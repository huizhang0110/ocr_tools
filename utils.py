import cv2
import xml.etree.ElementTree as ET


def crop_voc_image(xml_file, image_file, margin_h=None, margin_w=None):
    image = cv2.imread(image_file, 1)
    assert image is not None, "image file error: {}".format(image_file)
    img_h, img_w = image.shape[:2]

    image_list, name_list = [], []

    tree = ET.parse(xml_file)
    for obj in tree.findall("object"):
        name_list.append(obj.find("name").text)
        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        
        if margin_w is not None:
            xmin = max(xmin - margin_w, 0)
            xmax = min(xmax + margin_w, img_w)
        if margin_h is not None:
            ymin = max(ymin - margin_h, 0)
            ymax = min(ymax + margin_h, img_h)
    
        crop_image = image[ymin:ymax, xmin:xmax, :]
        image_list.append(crop_image)
    
    return image_list, name_list


