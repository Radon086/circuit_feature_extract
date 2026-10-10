import cv2
import numpy as np

"""
plan:

gray
  ↓
GaussianBlur
  ↓
Canny
  ↓
findContours
  ↓
approxPolyDP
  ↓
len(approx) == 4
  ↓
area filter
  ↓
convexity filter
  ↓
90° angle filter
  ↓
draw ALL candidates

"""

def before_contour_A(image_path):

    img = cv2.imread(image_path)
    print("Processing image:", image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    cv2.imshow("Gray Image", img_gray)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
#    cv2.imshow("Blurred Image", img_blur)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    _, img_thresh = cv2.threshold(
        img_blur, 200, 255, cv2.THRESH_BINARY_INV
    )
#    cv2.imshow("Adaptive Threshold Image", img_thresh)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    img_canny = cv2.Canny(img_thresh, 100, 200)
#    cv2.imshow("Canny Image", img_canny)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    cv2.imwrite("output/test_cv/Canny_Image_threshandcanny.png", img_canny)
    return img_canny

def before_contour_B(image_path):

    img = cv2.imread(image_path)
    print("Processing image:", image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 #   cv2.imshow("Gray Image", img_gray)
 #   cv2.waitKey(0)
 #   cv2.destroyAllWindows()
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
 #   cv2.imshow("Blurred Image", img_blur)
 #   cv2.waitKey(0)
 #   cv2.destroyAllWindows()
 #   cv2.destroyAllWindows()
 #   cv2.imshow("Canny Image", img_blur)
    img_canny = cv2.Canny(img_blur, 100, 200)
 #   cv2.imshow("Canny Image", img_canny)
 #   cv2.waitKey(0)
 #   cv2.destroyAllWindows()
    cv2.imwrite("output/test_cv/Canny_Image_nothresh.png", img_canny)
    return img_canny

def before_contour_C(image_path):

    img = cv2.imread(image_path)
    print("Processing image:", image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 #   cv2.imshow("Gray Image", img_gray)
 #   cv2.waitKey(0)
 #   cv2.destroyAllWindows()
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
 #   cv2.imshow("Blurred Image", img_blur)
 #   cv2.waitKey(0)
 #   cv2.destroyAllWindows()
    _, img_thresh = cv2.threshold(
        img_blur, 200, 255, cv2.THRESH_BINARY_INV
    )
#    cv2.imshow("Adaptive Threshold Image", img_thresh)
 #   cv2.waitKey(0)
#    cv2.destroyAllWindows()
    cv2.imwrite("output/test_cv/Canny_Image_thresh.png", img_thresh)
    return img_thresh

def find_contours(image, draw_contours=False, exp_name=None):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if draw_contours:
        img_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if exp_name is None:
            exp_name = ""
        id = 0
        filterd_contours = []
        epsilon_ratio = 0.02  # You can adjust this value based on your requirements
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            approx = cv2.approxPolyDP(cnt, epsilon_ratio * perimeter, True)

            #if len(approx) != 4:
            #    continue

            if not cv2.isContourConvex(approx):
                continue

            filterd_contours.append(approx)

        for contour in filterd_contours:
            tmp = img_contours.copy()
            cv2.drawContours(tmp, [contour], -1, (0, 255, 0), 2)
            cv2.imwrite(f"output/test_cv/Contours_Image_{exp_name}_{id}.png", tmp)
            id += 1
    return filterd_contours


def find_strokes(image_path, draw_contours=False, exp_name=None):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(
        gray, 200, 255, cv2.THRESH_BINARY_INV
    )

    dist = cv2.distanceTransform(
        binary, cv2.DIST_L2, 5
    )

    stroke_width = 2 * dist
    values = stroke_width[binary > 0]
    return values

def filter_by_thickness(image_path, threshold=4.0):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(
        gray, 200, 255, cv2.THRESH_BINARY_INV
    )

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_width = 2 * dist

    # Keep only pixels with estimated stroke width >= 4 px
    width_mask = np.zeros_like(binary)
    width_mask[stroke_width >= threshold] = 255

    gray_filtered = cv2.bitwise_and(cv2.bitwise_not(gray), width_mask)
    return width_mask, gray_filtered

def remove_small_areas(image, min_area=10):
    # Works with a binary image: background=0, foreground=255
    binary = np.where(image > 0, 255, 0).astype(np.uint8)

    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(binary, connectivity=8)

    filtered = np.zeros_like(binary)

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            filtered[labels == i] = 255

    return filtered

id =4
#img_path = ['./imgs/app_circuits/L298_8.png','./imgs/app_circuits/L298_9.png','./imgs/app_circuits/L298_10.png','./imgs/app_circuits/tps5430_7-9.png']
img_path = ['./imgs/app_circuits/L298_9_2.png']
for path in img_path:

    width_mask, _ = filter_by_thickness(path, threshold=5)
    #cv2.imwrite(f"output/test_cv/Width_Filtered_{id}.png", width_mask)
    #cv2.imwrite(f"output/test_cv/Gray_Filtered_{id}.png", gray_filtered)
    width_mask = remove_small_areas(width_mask, min_area=10)
    cv2.imwrite(f"output/test_cv/Width_Filtered_Small_Areas_{4}.png", width_mask)
    
    
    # kernel = np.ones((10,10),dtype=np.uint8)
    # processed = cv2.morphologyEx(gray_filtered, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite(f"output/test_cv/Processed_{id}.png", processed)
     


    ''''
    img_canny_A = before_contour_A(path)
    contours_A = find_contours(img_canny_A, draw_contours=True, exp_name="A")
    img_canny_B = before_contour_B(path)
    contours_B = find_contours(img_canny_B, draw_contours=True, exp_name="B")
    img_thresh_C = before_contour_C(path)
    contours_C = find_contours(img_thresh_C, draw_contours=True, exp_name="C")
    '''
    strokes = find_strokes(path, draw_contours=True, exp_name="strokes")
    print(strokes)

    print("Min:", strokes.min())
    print("Max:", strokes.max())
    print("Median:", np.median(strokes))
    print("90th percentile:", np.percentile(strokes, 90))
    print("95th percentile:", np.percentile(strokes, 95))

    print(np.min(strokes), np.max(strokes))
    id += 1
    