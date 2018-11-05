import cv2
import numpy as np


def crop_external_rect(ori_image, points):
    """Crop the smallest external rectangle according to the points
    """
    ori_h, ori_w = ori_image.shape[:2]
    points = np.array(points, dtype=np.int32)
    p_center, wh, angle = cv2.minAreaRect(points=points)
    M = cv2.getRotationMatrix2D(p_center, angle, 1.0)
    rot_image = cv2.warpAffine(ori_image, M, (ori_w, ori_h))

    p_w, p_h = np.array(p_center, np.int32)
    w, h = np.array(wh, np.int32)
    roi_image = rot_image[(p_h - h//2):(p_h + h - h//2), (p_w - w//2):(p_w + w - w//2), :]
    return roi_image



image = np.full(shape=(400, 400, 3), fill_value=255, dtype=np.uint8)
p0 = [40, 150]
p1 = [300, 50]
p2 = [350, 200]
p3 = [100, 360]
lines = np.array([[p0, p1, p2, p3]], dtype=np.int32)

cv2.polylines(image, lines, isClosed=True, color=(0, 255, 0), thickness=3)
cv2.fillPoly(image, lines, color=(0, 255, 0))

points = np.array([p0, p2, p3, p1])
crop_bbox(image, points)
# rect = cv2.minAreaRect(points=points)  # ((cx, cy), (w, h), angle)
# box_points = cv2.boxPoints(rect).astype(np.int32)
# cv2.polylines(image, [box_points], isClosed=True, color=(0, 0, 0), thickness=2)


