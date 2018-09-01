import xml.etree.ElementTree as ET
import os
import argparse
import cv2
from utils import crop_voc_image


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="")
parser.add_argument("--save_dir", type=str, help="")
parser.add_argument("--tags_file", type=str, help="")
parser.add_argument("--margin_w", type=int, default=2, help="")
parser.add_argument("--margin_h", type=int, default=2, help="")
args = parser.parse_args()


count = 0
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
tags_fo = open(args.tags_file, "w")

image_format = ["png", "jpg", "PNG", "JPG"]

for root, dirlist, filelist in os.walk(args.data_dir):
    for filename in filelist:
        if filename.endswith("xml"):
            xml_path = os.path.join(root, filename)
            for x in image_format:
                img_path = xml_path.replace("xml", x)
                if os.path.exists(img_path):
                    break
            try:
                image_list, name_list = crop_voc_image(xml_path, img_path, 
                        margin_w=args.margin_w, margin_h=args.margin_h)
            except Exception as e:
                print(e, ": xml_path <{}>".format(xml_path))
                continue
            for image, name in zip(image_list, name_list):
                save_path = os.path.join(args.save_dir, "{}.jpg".format(count))
                cv2.imwrite(save_path, image) 
                tags_fo.write("{} {}\n".format(os.path.abspath(save_path), name))
                count += 1
                if count % 10000 == 0:
                    print(count)

tags_fo.close()
print("Finished crop {} images, please check at {}".format(count, args.save_dir))
