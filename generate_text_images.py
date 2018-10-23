#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import pickle
from os.path import dirname
import math
from os.path import join, isdir, realpath
from os import makedirs
import codecs
import random
import argparse


DIR = dirname(os.path.realpath(__file__))
font_dir = 'fonts'

font_cache = os.path.join(font_dir, "fonts.pkl")

font_un = ['Kaiti.ttf', 'SimSun.ttf', 'SimHei.ttf', 'PMingLiU.ttf']
# pictures = os.listdir('pictures')

def add_salt_pepper(img, fill=(255, 255, 255), num_ratio=None):
    """
    Args:
        img: PIL.Image type
        fill: noise color, tuple of length 3, default: Write color
        num: noise spot number, int 
    Return:
        img: PIL.Image type
    """
    if num_ratio is None:
        num_ratio = np.random.uniform(0.06, 0.15)
    num = int(img.size[0] * img.size[1] * num_ratio)
    pix = img.load()
    for k in range(num):
        x = int(np.random.uniform(0, img.size[0]))
        y = int(np.random.uniform(0, img.size[1]))
        r, g, b = pix[x, y]
        img.putpixel([x, y], fill)
    return img


def refresh_fonts_cache():
    import glob
    from PIL import ImageFont
    en_fonts = []
    unicode_fonts = []
    for file in glob.glob(font_dir + "/*.ttf"):
        font = ImageFont.truetype(file, 14)
        name = os.path.basename(file)
        glyphs = font.font.glyphs
        print("name: %s, glyphs: %d, style: %s" % (font.font.family, glyphs, font.font.style))
        if glyphs > 10000:
            unicode_fonts.append(name)
        else:
            en_fonts.append(name)
    with open(font_cache, 'wb') as output:
        pickle.dump((en_fonts, unicode_fonts), output, pickle.HIGHEST_PROTOCOL)
    return en_fonts, unicode_fonts


if os.path.isfile(font_cache):
    with open(font_cache, 'rb') as f:
        en_fonts, unicode_fonts = pickle.load(f)
        print('fonts.pkl load down!')
else:
    en_fonts, unicode_fonts = refresh_fonts_cache()
    print('no pkl load')
all_fonts = en_fonts + unicode_fonts
# print('font:',unicode_fonts)

def has_unicode(text):
    for t in text:
        if ord(t) > 255:
            return True
    else:
        return False


def get_font(text, font_size, font_name=None, multi_fonts=False):
    if font_name is None and multi_fonts:
        if has_unicode(text):
            fonts = unicode_fonts
        else:
            fonts = all_fonts
        font_name = np.random.choice(fonts)
        while font_name == 'symbol.ttf':
            font_name = np.random.choice(fonts)
    else:
        font_name = unicode_fonts[0]
    font_file = os.path.join(font_dir, font_name)
    return ImageFont.truetype(font_file, font_size, encoding='unic')
    # return font_file


def split_text(text):
    if text is None:
        return []
    if len(text) == 1:
        return [(has_unicode(text), text)]
    start = None
    current_is_unicode = None
    texts = []
    for i in range(len(text)):
        is_unicode = ord(text[i]) > 255
        if current_is_unicode is None:
            current_is_unicode = is_unicode
            start = i
        elif current_is_unicode == is_unicode:
            pass
        else:
            texts.append((current_is_unicode, text[start:i]))
            start = i
            current_is_unicode = is_unicode
        if i == len(text)-1:
            texts.append((current_is_unicode, text[start:]))
    return texts

symbol_chn = ['℃','Ⅰ','Ⅱ','Ⅲ','Ⅳ','Ⅴ','Ⅵ','Ⅶ','Ⅷ','Ⅸ','Ⅹ','Ⅺ','Ⅻ','、',
'〃','〈','〉','《','》','【','】','＂','＃','＄','％','＆','＇','（','）','＊',
'＋','，','－','：','；','＜','＝','＞','？','^','～','°','￡','￥','”','“']

def need_rotate(text):
    if text is None:
        return []
    if len(text) == 1:
        if ord(text) < 255 or text in symbol_chn:
            return [(False,text)]
        else:
            return [(True,text)]
    start = None
    current_is_unicode = None
    texts = []
    for i in range(len(text)):
        is_unicode = ord(text[i]) > 255 and text[i] not in symbol_chn
        if current_is_unicode is None:
            current_is_unicode = is_unicode
            start = i
        elif current_is_unicode == is_unicode:
            pass
        else:
            texts.append((current_is_unicode, text[start:i]))
            start = i
            current_is_unicode = is_unicode
        if i == len(text)-1:
            texts.append((current_is_unicode, text[start:]))
    return texts


def random_noise(image, mode=None):
    from skimage.util import random_noise, img_as_float, img_as_ubyte
    modes = ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']
    if mode is None:
        mode = np.random.choice(modes)
    image = img_as_float(image)
    image = random_noise(image, mode)
    image = img_as_ubyte(image)
    return image

def dither(num, thresh=127):
    derr = np.zeros(num.shape, dtype=int)

    div = 8
    for y in range(num.shape[0]):
        for x in range(num.shape[1]):
            newval = derr[y, x] + num[y, x]
            if newval >= thresh:
                errval = newval - 255
                num[y, x] = 1.
            else:
                errval = newval
                num[y, x] = 0.
            if x + 1 < num.shape[1]:
                derr[y, x + 1] += errval / div
                if x + 2 < num.shape[1]:
                    derr[y, x + 2] += errval / div
            if y + 1 < num.shape[0]:
                derr[y + 1, x - 1] += errval / div
                derr[y + 1, x] += errval / div
                if y + 2 < num.shape[0]:
                    derr[y + 2, x] += errval / div
                if x + 1 < num.shape[1]:
                    derr[y + 1, x + 1] += errval / div
    return num[::-1, :] * 255


def hyperdither(img):
    bottom = False
    if bottom:
        m = np.array(img)[::-1, :]
        m2 = dither(m, thresh=127)
        out = Image.fromarray(m2[:, :])
    else:
        m = np.array(img)[:, :]
        m2 = dither(m, thresh=127)
        out = Image.fromarray(m2[::-1, :])
    return out


def paint_vertical_text(text, size=None, font_size=None, shift=False, rotate=False, multi_fonts=False,
                        random_color=False, font_name=None):
    if font_size is None:
        font_size = np.random.randint(28, 32)
    if random_color:
        if np.random.randint(0, 100) < 5:
            bg_color_r = np.random.randint(0, 255)
            bg_color_g = np.random.randint(0, 255)
            bg_color_b = np.random.randint(0, 255)
            text_color_r = (bg_color_r + 100) % 255
            text_color_g = (bg_color_g + 100) % 255
            text_color_b = (bg_color_b + 100) % 255
        else:
            bg_color_r = bg_color_g = bg_color_b = 255
            text_color_r = text_color_g = text_color_b = np.random.randint(0, 100)
            # color = np.random.randint(0, 100)
    else:
        bg_color_r = bg_color_g = bg_color_b = 255
        text_color_r = text_color_g = text_color_b = 0
    font_name = np.random.choice(font_un)
    font_file = os.path.join(font_dir, font_name)
    # font = ImageFont.truetype(font_file, font_size, encoding='unic')
    # font_file = get_font(text, font_size, font_name, multi_fonts)
    # texts = split_text(text)
    texts = need_rotate(text)
    def text_image(text):
        font = ImageFont.truetype(font_file, font_size, encoding='unic')
        text_w, text_h = font.getsize(text)
        # image = Image.new('L', (text_w, text_h), color=bg_color)
        image = Image.new('RGB', (text_w, text_h), color=(bg_color_r, bg_color_g, bg_color_b))
        draw = ImageDraw.Draw(image)
        # 干扰线
        if np.random.randint(0, 100)<0:
            line_count = np.random.randint(1, 4)
            for i in range(line_count):
                x1 = np.random.randint(0, image.height)
                x2 = np.random.randint(0, image.height)
                y1 = np.random.randint(0, image.width)
                y2 = np.random.randint(0, image.width)
                line_width = np.random.randint(1, 5)
                fill_color_b = np.random.randint(0, 255)
                fill_color_g = np.random.randint(0, 255)
                fill_color_r = np.random.randint(0, 255)
                draw.line([(x1, y1), (x2, y2)], fill=(fill_color_b, fill_color_g, fill_color_r), width=line_width)
        # 干扰背景
        if np.random.randint(0, 100)<5:
            x1 = np.random.randint(0, image.height)
            x2 = np.random.randint(0, image.height)
            y1 = np.random.randint(0, image.width)
            y2 = np.random.randint(0, image.width)
            line_width = np.random.randint(image.height/2, image.height)
            fill_color_r = (text_color_r + 80)
            fill_color_g = (text_color_g + 80)
            fill_color_b = (text_color_b + 80)
            draw.line([(x1, y1), (x2, y2)], fill=(fill_color_b, fill_color_g, fill_color_r), width=line_width)
        draw.text((0, 0), text, font=font, fill=(text_color_r, text_color_g, text_color_b))
        return image

    word_images = []
    for is_unicode, word in texts:
        if is_unicode:
            for char in word:
                word_image = text_image(char)
                word_images.append(word_image)
        else:
            word_image = text_image(word)
            # print('word_image: ',word_image.size[0],word_image.size[1])
            # cv2.imshow('word_image', np.array(word_image))
            # cv2.waitKey()
            word_image = word_image.rotate(-90, expand=True)
            word_images.append(word_image)
            # return
    if word_images==[]:
        # print('not chn')
        return
    random_b = random.random()
    if random_b > 0.4:
        border = 0
    elif random_b <= 0.4 and random_b > 0.2:
        border = 1
    else:
        border = 2
    random_c = random.random()

    if random_c > 0.4:
        compact = 0
    elif random_c <= 0.4 and random_c > 0.3:
        compact = 3
    else:
        compact = 5
    # compact = 1
    # border = 0#np.random.randint(0, 1)
    text_w = max([img.size[0] for img in word_images])
    text_h = sum([img.size[1] for img in word_images]) + compact*(len(word_images)-1)
    w = text_w + border * 2
    h = text_h + border * 2# + np.random.randint(0, 8)
    # image = Image.new('L', (w, h), color=bg_color)
    image = Image.new('RGB', (w, h), color=(bg_color_r, bg_color_g, bg_color_b))
    if shift:
        max_shift = max(0, (h - text_h) // 2)
        y = (h - text_h) // 2 + np.random.randint(-max_shift, max_shift+1)
    else:
        y = (h - text_h) // 2
    for img in word_images:
        iw, ih = img.size
        # print(iw,ih)
        # cv2.imshow('img', np.array(img))
        # cv2.waitKey()
        image.paste(img, ((w-iw)//2, y))
        y = y + ih + compact
    if rotate:
        angle = np.random.randint(-3, 4)
        image = image.convert('RGBA')
        # if np.random.randint(0,100)>50:
        if 100>50:
            rot = image.rotate(90+angle, resample=Image.BICUBIC, expand=True)
        else:
            rot = image.rotate(angle, resample=Image.BICUBIC, expand=True)
        fff = Image.new('RGBA', rot.size, (255,)*4)
        # fff = Image.new('RGBA', rot.size, (bg_color, bg_color, bg_color, 255))
        # image = Image.composite(rot, fff, rot).convert("L")
        image = Image.composite(rot, fff, rot).convert("RGB")
    else:
        image = image.rotate(90, expand=True)
    if random.random() > 0.9:
        image = image.resize((image.size[0]*9//10,image.size[1]),Image.ANTIALIAS)
    return image


def paint_text(text, size=None, font_size=None, shift=False, rotate=False, multi_fonts=False,
               random_color=False, font_name=None):
    if font_size is None:
        if np.random.randint(0, 100) < 50:
            font_size = np.random.randint(12, 20)
        else:
            font_size = np.random.randint(20, 32)

    font = get_font(text, font_size, font_name, multi_fonts)

    offset_x, offset_y = font.getoffset(text)
    text_w, text_h = font.getsize(text)
    if text_w == 0 or text_h == 0:
        return None
    if size is None:
        border = np.random.randint(0, 3)
        if np.random.randint(0, 100) < 70:
            w = text_w + border * 2 + np.random.randint(0, 2)
        else:
            w = text_w + border * 2 + np.random.randint(40, 60)
        h = text_h + border * 2
    else:
        w, h = size

    if shift:
        max_shift = max(0, min(5, (w - text_w) // 2, (h - text_h) // 2))
        x = (w - text_w) // 2 + np.random.randint(-max_shift, max_shift+1)
        y = (h - text_h) // 2 + np.random.randint(-max_shift, max_shift+1) - offset_y // 2
    else:
        x = (w - text_w) // 2 
        y = (h - text_h) // 2 - offset_y // 2
    if random_color:
        if np.random.randint(0, 100) < 70:
            if np.random.randint(0, 100) > 50:
                #灰底灰字
                bg_color = np.random.randint(156, 256)
                text_color = np.random.randint(0, bg_color - 100)
            else:
                if np.random.randint(0, 100) > 50:
                    #黑底白字
                    bg_color = np.random.randint(0, 100)
                    text_color = np.random.randint(min(bg_color+80,254), 255)
                else:
                    #灰底白字
                    bg_color = np.random.randint(100, 180)
                    text_color = np.random.randint(min(bg_color+80,254), 255)
            # bg_color = np.random.randint(0, 255)
            # text_color = (bg_color + 60) % 255
        else:
            #白底灰字
            bg_color = 255
            text_color = np.random.randint(0, 100)
            # color = np.random.randint(0, 100)
    else:
        bg_color = 255
        text_color = 0
    image = Image.new('L', (w, h), color=bg_color)
    (image_w, image_h) = image.size
    # image = Image.new('RGB', (w, h), color=(bg_color_r, bg_color_g, bg_color_b))
    draw = ImageDraw.Draw(image)
    # 干扰线
    if len(text) > 1 and np.random.randint(0, 100)<10:
        line_count = np.random.randint(1, 4)
        for i in range(line_count):
            x1 = np.random.randint(0, image.width)
            x2 = np.random.randint(0, image.width)
            y1 = np.random.randint(0, image.height)
            y2 = np.random.randint(0, image.height)
            line_width = np.random.randint(1, 5)
            fill_color = np.random.randint(0, 255)

            draw.line([(x1, y1), (x2, y2)], fill=fill_color, width=line_width)
    # 干扰背景
    if len(text) > 1 and np.random.randint(0, 100)<1:
        x1 = np.random.randint(0, image.width)
        x2 = np.random.randint(0, image.width)
        y1 = np.random.randint(0, image.height)
        y2 = np.random.randint(0, image.height)
        line_width = np.random.randint(image.height/2, image.height)
        fill_color = (text_color + 80)%255
        draw.line([(x1, y1), (x2, y2)], fill=fill_color, width=line_width)

    draw.text((x, y), text, font=font, fill=text_color)
    # 加入白色的椒盐噪声
    if np.random.randint(0, 100) < 0:
        add_salt_pepper(image)
    # 加入黑色的椒盐噪声
    if np.random.randint(0, 100) < 0:
        add_salt_pepper(image, (0, 0, 0))
    #是否添加高斯噪声或扫描件噪声
    if np.random.randint(0, 100) < 40:
        # 添加高斯背景噪声
        if np.random.randint(0, 100) < 30:
            background_gauss = np.ones((h, w)) * 255
            cv2.randn(background_gauss, 235, 10)
            background_gauss=Image.fromarray(background_gauss).convert('L')
            mask = image.point(lambda x: 0 if x == 255 or x == 0 else 255, '1')
            background_gauss.paste(image, (0, 0), mask=mask)
            image = background_gauss
        #添加扫描件抖动噪声
        else:
            if font_size > 20:
                image = hyperdither(image)
    #加入背景图片
#     if np.random.randint(0, 100) < 0:
        # assert len(pictures) > 0,"no picture!"
        # picture = Image.open('pictures/' + pictures[random.randint(0, len(pictures) - 1)])
        # if picture.size[0] < w:
            # picture = picture.resize([w, int(picture.size[1] * (w / picture.size[0]))], Image.ANTIALIAS)
        # elif picture.size[1] < h:
            # picture.resize([int(picture.size[0] * (h / picture.size[1])), h], Image.ANTIALIAS)
        # x = random.randint(0, picture.size[0] - w)
        # y = random.randint(0, picture.size[1] - h)
        # background_pic = picture.crop((x,y,x + w,y + h))
        # mask = image.point(lambda x: 0 if x == 255 or x == 0 else 255, '1')
        # background_pic.paste(image, (0, 0), mask=mask)
#         image = background_pic
    if rotate:
        if np.random.randint(0, 100) < 20:
            angle = np.random.randint(-2, 3)
        else:
            angle = 0
        # if np.random.randint(0, 100)<50:
        #     angle = 45
        # else:
        #     angle = -45
        image = image.convert('RGBA')
        rot = image.rotate(angle, resample=Image.BICUBIC, expand=True)
        fff = Image.new('RGBA', rot.size, (255,)*4)
        # fff = Image.new('RGBA', rot.size, (bg_color, bg_color, bg_color, 255))
        # image = Image.composite(rot, fff, rot).convert("L")
        image = Image.composite(rot, fff, rot).convert("L")
    # cv2.imshow('img', np.array(image))
    # cv2.waitKey()
    #字体压缩
    if np.random.randint(0, 100) < 30:
        temp = np.random.randint(0, 100)
        if temp <= 35 and font_size > 16: 
            image = image.resize((image.size[0]*6//10,image.size[1]),Image.ANTIALIAS)
        if temp > 35 and temp <= 50:
            image = image.resize((image.size[0]*14//10,image.size[1]),Image.ANTIALIAS)
        if temp > 50 and temp <= 85 and font_size > 16:
            image = image.resize((image.size[0],image.size[1]*7//10),Image.ANTIALIAS)
        if temp > 85:
            image = image.resize((image.size[0],image.size[1]*14//10),Image.ANTIALIAS)
    return image


def random_pad_image(image):
    max_width = image.shape[-1] + np.random.randint(1, 64)
    max_width = min(512, max_width)
    if max_width >= image.shape[-1]:
        return image
    padded_data = np.zeros((1, image.shape[-2], max_width), dtype=np.uint8)
    padded_data.fill(255)
    # x = (self.max_width - data.shape[-1])//2
    x = np.random.randint(0, max_width-image.shape[-1])
    padded_data[0, 0:image.shape[-2], x:x+image.shape[-1]] = image
    return padded_data


def normalize_image(img):
    w, h = img.size
    aspect_ratio = float(w) / float(h)
    # if aspect_ratio < float(self.bucket_min_width) / self.image_height:
    #     img = img.resize(
    #         (self.bucket_min_width, self.image_height),
    #         Image.ANTIALIAS)
    # elif aspect_ratio > float(
    #         self.bucket_max_width) / self.image_height:
    #     img = img.resize(
    #         (self.bucket_max_width, self.image_height),
    #         Image.ANTIALIAS)
    # elif h != self.image_height:
    #     img = img.resize(
    #         (int(aspect_ratio * self.image_height), self.image_height),
    #         Image.ANTIALIAS)

    # 图是竖直的，即长边是高，需要归一到宽为image_height
    if aspect_ratio < 1:
        img = img.resize(
            (32, int(32/aspect_ratio)),
            Image.ANTIALIAS)
        w_new, h_new = img.size
        if h_new<12:
            img = img.resize((
                32, 12),
                Image.ANTIALIAS)
        elif h_new>512:
            img = img.resize((
                32, 512),
                Image.ANTIALIAS)
    # 图水平的，即长边是宽,需要归一到高为image_height
    else:
        img = img.resize(
            (int(32*aspect_ratio), 32),
            Image.ANTIALIAS)
        w_new, h_new = img.size
        if w_new<12:
            img = img.resize(
                (12, 32),
                Image.ANTIALIAS)
        elif w_new>512:
            img = img.resize(
                (512, 32),
                Image.ANTIALIAS)
    w, h = img.size
    # if w==32 and h!=32:
    #     img = img.rotate(90, resample=Image.BICUBIC, expand=True)
    # img_bw = img.convert('L')
    img_bw = np.asarray(img, dtype=np.uint8)
    # img_bw = img_bw.transpose([2, 0, 1])
    # img_bw = img_bw[np.newaxis, :]
    return img_bw


def test_paint_text():
    for i in range(1000):
        # text = np.random.choice(words.CHARS)
        text = u'（三）电新杉杉股份'
        if len(text) > 2 and np.random.randint(0, 100) > 50:
            image = paint_vertical_text(text, size=None, shift=True, rotate=True, multi_fonts=True, random_color=True)
            random_invert = False
        else:
            # image = paint_text(text, size=None, shift=True, rotate=True, multi_fonts=True, random_color=True)
            image = paint_text(text, size=None, shift=True, rotate=True, multi_fonts=True, random_color=True)
            random_invert = False
        image = normalize_image(image)
        image = np.array(image)
        image = random_noise(image)
        if len(text) >= 3 and np.random.randint(0, 100) > 70:
            image = cv2.flip(image, -1)
        if np.random.randint(0, 100) > 50:
            image = random_noise(image)
        if random_invert and np.random.randint(0, 100) > 70:
            image = 255 - image
        if np.random.randint(0, 100)>30:
            image = random_pad_image(image)

        cv2.imwrite(join(DIR, 'gen_images/%d.jpg' % i), image)
        # cv2.imshow('image', image)
        # cv2.waitKey()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_list_file", type=str, help="", default="./word.txt")
    parser.add_argument("--tags_file", type=str, default="output.tags", help="")
    parser.add_argument("--save_dir", type=str, default="output", help="")
    parser.add_argument("--epoch", type=int, default=3, help="")
    args = parser.parse_args()
    
    gen_images_dir = args.save_dir
    words_file = args.word_list_file
    tags_file = args.tags_file

    if not os.path.exists(gen_images_dir):
        os.makedirs(gen_images_dir)
    image_id = 1
    word_list = list()

    def read_words_file(words_file):
        with codecs.open(words_file, 'r', encoding='utf-8') as the_file:
            if words_file.endswith("tags"):
                lines = [line.split(" ", 1)[-1] for line in the_file.readlines()]
            else:
                lines = the_file.readlines()

            for line in lines:
                line = line.strip()
                if len(line) > 0:
                    word_list.append(line)

    read_words_file(words_file)
    # word_list = sorted(list(word_list))

    print('len of word list = %d' % (len(word_list), )) # 141036

    def generate_image(tags_fobj, text, image_id):
        image = paint_text(text, size=None, shift=True, rotate=True, multi_fonts=True, random_color=True)
        if image is None:
            print("paint_text: %s failed" % text)
        image_path = join(gen_images_dir, "%d.jpg" % image_id)
        image = image.convert('L')
        image.save(image_path, quality = np.random.randint(80, 100))
        tags_fobj.write('%s %s\n' % (image_path, text))
        return image

    print('start generating images')
    with codecs.open(tags_file, 'w', encoding='utf-8') as the_file:
        for num_ in range(args.epoch):  # 每个word生成n张图片
            for text in word_list:
                generate_image(the_file, text, image_id)
                image_id += 1
                if image_id % 1000 == 0:
                    print('image id: ', image_id)
                # if image_id == 200:
                #     break
    print("All finished")

