import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("--tags_file", type=str, help="")
parser.add_argument("--out_file", type=str, default="sorted.tags", help="")
args = parser.parse_args()


line_width_dict = {}

with open(args.tags_file) as fo:
    for line in fo:
        image_path = line.split(" ", 1)[0]
        image_h, image_w = cv2.imread(image_path, 1).shape[:2]
        new_w = 32 * image_w / image_h
        if new_w > 600:
            continue
        line_width_dict[line] = new_w

sorted_ = sorted(line_width_dict.items(), key=lambda item : item[1])

with open(args.out_file, "w") as fo:
    lines = [x[0] for x in sorted_]
    fo.write("".join(lines))
print("Finished write sorted tags file, please check at {}".format(args.out_file))

