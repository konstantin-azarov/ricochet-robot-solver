import cv2
import math
import numpy as np
import sys

width = 1664
height = 936

def sweep_line(img, angle, bucket_width, threshold):
    ofs = []
    k = -math.tan(angle)
    for y in xrange(height):
        ofs.append(int(round(y * k)))

    left_sum = [0] * height
    right_sum = [0] * height

    min_x = max(0, -min(ofs)) + bucket_width
    max_x = min(width - 1, width - max(ofs)) - bucket_width

    best_x = 0
    max_score = -1

    for x in xrange(min_x, max_x):
        score = 0
        for y in xrange(height):
            x0 = ofs[y] + x
            left_sum = 0.0
            right_sum = 0.0
            for xx in xrange(x0 - bucket_width + 1, x0 + 1):
                left_sum += img[y, xx]
            for xx in xrange(x0 + 1, x0 + bucket_width + 1):
                right_sum += img[y, xx]

            if (right_sum - left_sum) / bucket_width > threshold:
                score += 1

        if score > max_score:
            max_score = score
            best_x = x

    return best_x, max_score
    

img_bgr = cv2.resize(
    cv2.imread(sys.argv[1]),
    (1664, 936))

img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

img_mask = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([255, 70, 200]))

# x0, score =  sweep_line(img_gray, math.pi/180*6, 5, 20)

# cv2.line(img_bgr, (x0, 0), (int(x0 - math.tan(math.pi/180*7.5) * height), height), (0, 0, 255))


kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
img_mask = cv2.dilate(img_mask, kernel, iterations = 3)
cc_count, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(img_mask) 

cv2.imshow("debug", img_hsv)

max_size = 0
cc_label = 0
for i, cc in zip(range(cc_count), cc_stats[1:]):
    if cc[cv2.CC_STAT_AREA] > max_size:
        max_size = cc[cv2.CC_STAT_AREA]
        cc_label = i+1

img_mask = ((cc_labels == cc_label) * 255).astype("uint8")

img_mask, contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour = contours[0].reshape((-1, 2))
hull = cv2.convexHull(contour)

img_dbg = cv2.drawContours(img_bgr, [hull], -1, color=(255, 0, 0))

edges = cv2.Canny(cv2.blur(img_gray, (3,3)), 50, 150, 3)


cv2.imshow("debug2", img_bgr)
cv2.waitKey(-1)
