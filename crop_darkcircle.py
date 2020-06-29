import cv2
import dlib
import numpy as np
import os


predictor_model = r'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_model)
imgs_path = r"A:\picture\blackeye\0624\0"
result_path = r'A:\picture\blackeye\dark_circle\0'
for img_path in os.listdir(imgs_path):
    img = cv2.imread(os.path.join(imgs_path, img_path))
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except BaseException:
        print("cvtColor报错图片为：", os.path.join(imgs_path, img_path))
    rects = detector(img_gray, 0)
    try:
        detect = predictor(img, rects[0])
    except BaseException:
        print("detect报错图片为：", os.path.join(imgs_path, img_path))
    try:
        landmarks = np.matrix([[p.x, p.y] for p in detect.parts()])
        landmarks[landmarks < 0] = 0
    except BaseException:
        print("landmarks报错图片为：", os.path.join(imgs_path, img_path))
    lmrks = landmarks.A
    x0 = (lmrks[0][0] + lmrks[17][0] + lmrks[36][0])//3
    y0 = (lmrks[17][1] + lmrks[21][1])//2
    x1 = (lmrks[39][0] + lmrks[21][0] + lmrks[27][0])//3
    y1 = (lmrks[30][1] + lmrks[1][1])//2
    try:
        cropped1 = img[y0:y1, x0:x1]
    except BaseException:
        print("crop1报错图片为：", os.path.join(imgs_path, img_path))
    x2 = (lmrks[22][0] + lmrks[27][0] + lmrks[42][0])//3
    y2 = (lmrks[26][1] + lmrks[22][1])//2
    x3 = (lmrks[45][0] + lmrks[26][0] + lmrks[16][0])//3
    y3 = (lmrks[30][1] + lmrks[15][1])//2
    cropped2 = img[y2:y3, x2:x3]
    # # cv2.imshow("img1", cropped1)
    # cv2.imshow("img", cropped2)
    # cv2.waitKey(0)
    try:
        cv2.imwrite(os.path.join(result_path, "left1_{}".format(img_path)), cropped1)
        cv2.imwrite(os.path.join(result_path, "right1_{}".format(img_path)), cropped2)
    except BaseException:
        print("imwrite报错图片为：", os.path.join(imgs_path, img_path))












