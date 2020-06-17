import cv2
import numpy as np

cap = cv2.VideoCapture(0)   #lấy camera 0

while True:
    ret, vid = cap.read()
    gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    filt = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(filt, 50, 150)
    #ret, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)
    linesP = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, minLineLength=10,maxLineGap=10)
    if linesP is not None:
        for line in linesP:
            for x1, y1, x2, y2 in line:
                cv2.line(vid, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.imshow('Hough', vid)

    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

cap.release()
cv2.destroyAllWindows()