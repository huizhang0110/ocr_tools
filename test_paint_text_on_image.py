from PIL import Image, ImageDraw, ImageFont


filename = "font_posn_test.png"
fontname = "./fonts/Arial.ttf"
textsize = 12
text = "this is a test, sentense"

image_w = 300
image_h = 32
image = Image.new("RGBA", (image_w, image_h))
draw = ImageDraw.Draw(image)
font = ImageFont.truetype(fontname, textsize)

offset_y = font.getoffset(text)[1]
font_width, font_height = font.getsize(text)

print(image_h)  # 32
print(font_height) # 34
print(offset_y)

tmp = (image_h - font_height) // 2

draw.text((0, tmp), text, font=font, fill="red")

image.show()
