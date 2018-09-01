import cv2
import numpy as np
import math


def general_crop(image, tile, reverse_tile=True, margin_ratio=None):
    """Crop the image giving a tile.
    """
    if reverse_tile:
        tile[1:] = tile[::-1][:3]  # 调整点的顺序
    x = [p[0] for p in tile]
    y = [p[1] for p in tile]
    # phase1:shift the center of patch to image center
    x_center = int(round(sum(x) / 4))
    y_center = int(round(sum(y) / 4))
    im_center = [int(round(coord / 2)) for coord in image.shape[:2]]
    shift = [im_center[0] - y_center, im_center[1] - x_center]
    M = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
    height, width = image.shape[:2]
    im_shift = cv2.warpAffine(image, M, (width, height))
    cv2.imshow("shift", im_shift)
    cv2.waitKey(0)

    # phase2:imrote the im_shift to regular the box
    bb_width = max(math.sqrt((y[1] - y[0]) ** 2 + (x[1] - x[0]) ** 2),
                    math.sqrt((y[3] - y[2]) ** 2 + (x[3] - x[2]) ** 2))
    bb_height = max(math.sqrt((y[3] - y[0]) ** 2 + (x[3] - x[0]) ** 2),
                    math.sqrt((y[2] - y[1]) ** 2 + (x[2] - x[1]) ** 2))

    if bb_width > bb_height:  # main direction is horizental
        tan = ((y[1] - y[0]) / float(x[1] - x[0] + 1e-8) +
               (y[2] - y[3]) / float(x[2] - x[3] + 1e-8)) / 2
        degree = math.atan(tan) / math.pi * 180
    else:  # main direction is vertical
        tan = ((y[1] - y[2]) / float(x[1] - x[2] + 1e-8) +
               (y[0] - y[3]) / float(x[0] - x[3] + 1e-8)) / 2
        # degree = 90 + math.atan(tan) / math.pi * 180
        degree = math.atan(tan) / math.pi * 180 - np.sign(tan) * 90

    rotation_matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), degree, 1)
    im_rotate = cv2.warpAffine(im_shift, rotation_matrix, (width, height))
    cv2.imshow("rotate", im_rotate)
    cv2.waitKey(0)
    # phase3:crop the box out.
    x_min = im_center[1] - int(round(bb_width / 2))
    x_max = im_center[1] + int(round(bb_width / 2))
    y_min = im_center[0] - int(round(bb_height / 2))
    y_max = im_center[0] + int(round(bb_height / 2))
    # phase4: add some margin
    if margin_ratio is not None:
        margin_x = int(round((x_max - x_min) * margin_ratio / 2))
        margin_y = int(round((y_max - y_min) * margin_ratio / 2))
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(width, x_max + margin_x)
        y_max = min(height, y_max + margin_y)
    return im_rotate[y_min:y_max, x_min:x_max, :]


image = np.full(shape=(400, 400, 3), fill_value=255, dtype=np.uint8)
p0 = [40, 150]
p1 = [300, 50]
p2 = [350, 200]
p3 = [100, 360]

lines = np.array([[p0, p1, p2, p3]], dtype=np.int32)
# tile = [p0, p1, p2, p3]
# tile = [p3, p2, p1, p0]
# tile = [p0, p3, p2, p1]
# tile = [p1, p0, p3, p2]
tile = [p2, p3, p0, p1]


cv2.polylines(image, lines, isClosed=True, color=(0, 255, 0), thickness=3)
cv2.fillPoly(image, lines, color=(0, 0, 255))
image = general_crop(image, tile, reverse_tile=False)

cv2.imshow("crop_final", image)
cv2.waitKey()

