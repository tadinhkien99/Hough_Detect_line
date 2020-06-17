import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

def nothing(x):
    pass

#img = cv2.imread('street_test.jpg')                 # đọc file ảnh
img = cv2.imread(args["image"])
img = cv2.pyrUp(img)                                # tăng độ phân giải ảnh
img = cv2.resize(img, (980,612))                    # resize ảnh thành một size cố định
img_cp = img.copy()                                 # tạo 1 ảnh copy từ ảnh img
img_cpp = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # convert ảnh sang ảnh xám để thực hiện lọc Gausian tìm cạnh

def nothing(x):                                     # tạo hàm rỗng
    pass
def filter(gray):
    kernel = np.ones((5, 5), np.float32) / 25 # tạo mask với các pixel = 1 sau đó chia 25
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    filter_2D = cv2.filter2D(gray,-1, kernel=kernel)
    blur = cv2.blur(gray, (5, 5))
    median = cv2.medianBlur(gray,5)
    bila = cv2.bilateralFilter(gray,9,75,75)

    fig = plt.figure()                                         # hiển thị hình ảnh dùng matplotlib
    a = fig.add_subplot(3, 2, 1)
    a.set_title('Gaussian Blur')
    imgplot = plt.imshow(gauss, cmap='gray', vmin=0, vmax=255) # hiển thị hình ảnh dạng gray scale
    plt.axis("off")                                            # Không hiện trục
    a = fig.add_subplot(3, 2, 2)
    a.set_title('Filter 2D')
    imgplot = plt.imshow(filter_2D, cmap='gray', vmin=0, vmax=255)
    plt.axis("off")
    a = fig.add_subplot(3, 2, 3)
    a.set_title('Blur')
    imgplot = plt.imshow(blur, cmap='gray', vmin=0, vmax=255)
    plt.axis("off")
    a = fig.add_subplot(3, 2, 4)
    a.set_title('Median Blur')
    imgplot = plt.imshow(median, cmap='gray', vmin=0, vmax=255)
    plt.axis("off")
    a = fig.add_subplot(3, 2, 5)
    a.set_title('Bilateral Filter')
    imgplot = plt.imshow(bila, cmap='gray', vmin=0, vmax=255)
    plt.axis("off")
    plt.show()

def CannyTrackbar(gray):                            # HÀM PHÁT HIỆN CẠNH CANNY
    win_name = "Canny"                              # đặt tên cửa sổ trackbar
    cv2.namedWindow(win_name)
    cv2.createTrackbar("canny_high", win_name, 0, 255, nothing) # tạo trackbar mức high của bộ lọc canny
    cv2.createTrackbar("canny_low", win_name, 0, 255, nothing)  # tạo trackbar mức low của bộ lọc canny

    while True:
        low = cv2.getTrackbarPos("canny_low", win_name)
        high = cv2.getTrackbarPos("canny_high", win_name)
        filt = cv2.GaussianBlur(gray, (5, 5), 0)                 # dùng bộ lọc Gausian làm mịn ảnh

        canny = cv2.Canny(filt, low, high)                       # tìm cạnh của ảnh với canny
        cv2.imshow(win_name, canny)                              # hiển thị ảnh sau khi đã được phát hiện cạnh

        if cv2.waitKey(1) & 0xFF == ord('b'):                    # chờ tới khi ấn phím 'b' trên bàn phím
            break                                                # thì thoát khỏi vòng lặp while

    cv2.destroyAllWindows()                                      # đóng tất cả cửa sổ đang mở
    return canny

def hough_line(canny):                                      # HÀM BIẾN ĐỔI HOUGH THƯỜNG
    lines = cv2.HoughLines(canny, 1, np.pi/180, 150)        # biến đổi hough trên ảnh đã phát hiện cạnh
    if lines is not None:
        for line in lines:
            for rho, theta in line:         # rho, theta đại diện cho hệ số ro và góc theta trong công thức
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho                # theo công thức cạnh góc vuông bằng cạnh huyền nhân cos góc kề
                y0 = b * rho                # theo công thức cạnh góc vuông bằng cạnh huyền nhân sin góc đối
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img_cp, (x1, y1), (x2, y2), (203, 192, 255), 1)    # vẽ đường thẳng qua 2 điểm
    return img_cp

def hough_lineP(canny):             # HÀM HOUGH Probabilistic
    linesP = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, minLineLength=100,maxLineGap=50)    #minLineLength là độ dài nhỏ nhất tìm thấy được coi là cạnh
    if linesP is not None:                                                                  #maxLineGap là khoảng cách tối đa giữa 2 điểm trên cùng 1 đường thẳng để nối lại
        for line in linesP:                                                                 # 50 ở giữa là ngưỡng, >50 mới lấy
            for x1, y1, x2, y2 in line:    # HoughLineP trả về giá trị x1, y1, x2, y2 nên ta không cần tính 4 giá trị này
                cv2.line(img_cpp, (x1, y1), (x2, y2), (255, 100, 0), 2)
    print(line)
    return img_cpp

filt = filter(gray)
canny = CannyTrackbar(gray)
#cn_cp = canny.copy()
hough = hough_line(canny)
houghp = hough_lineP(canny)
cv2.imshow("canny_edge", canny)
cv2.imshow("hough", hough)
cv2.imshow("hough_p", houghp)
cv2.waitKey(0)

