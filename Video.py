import cv2
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", required=True,
	help="path to input video")
args = vars(ap.parse_args())

#cap = cv2.VideoCapture('road_test.mp4') # load video
cap = cv2.VideoCapture(args["video"])
frame_width = int(cap.get(3))           # lấy chiều rộng frame video
frame_height = int(cap.get(4))          # lấy chiều cao frame video
frame_fps = int(cap.get(5))             # lấy fps (frame per second) video
size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')    # FourCC là mã 4 byte được sử dụng để chỉ định codec video
out = cv2.VideoWriter('Hough.avi', fourcc, frame_fps, size, True)   # xuất video


def canny_edge_detector(vid):
    # Convert the vid color to grayscale
    gray_vid = cv2.cvtColor(vid, cv2.COLOR_RGB2GRAY)

    # Reduce noise from the video
    filt = cv2.GaussianBlur(gray_vid, (5, 5), 0)
    canny = cv2.Canny(filt, 50, 150)
    return canny

def region(vid):
    height = vid.shape[0] # lấy chiều cao video
    #polygons = np.array([[(300, height), (880, height), (550, 300)]])
    polygons = np.array([[(400, height), (900, height), (550, 300), (400, 300)]]) # tạo một đa giác để cắt vùng dưới video (vì ta chỉ detect làn đường)
    mask = np.zeros_like(vid) # tạo mặt nạ với tất cả pixel = 0

    # Fill poly-function deals with multiple polygon
    cv2.fillPoly(mask, polygons, 255)   # tạo mask với tất cả pixel = 1 theo độ rộng đa giác

    # Bitwise operation between canny vid and mask vid
    masked_vid = cv2.bitwise_and(vid, mask) # and bit mask với video (1 and 1 = 1, 1 and 0 = 0), chỉ giữ lại các cạnh có thể là làn đường
    return masked_vid

def create_coordinates(vid, line_parameters): #HÀM TẠO TỌA ĐỘ 4 ĐIỂM
    try:
        slope, intercept = line_parameters  # kiểm tra nếu không có slope và intercept trả về
    except TypeError:                       # thì mặc định
        slope, intercept = 0.000001, 0      # slope = 0.000001 và intercept = 0
    y1 = vid.shape[0]   # lấy chiều cao của video
    y2 = int(y1 -145)   # lấy độ dài tùy ý (145)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(vid, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            # x1, y1, x2, y2 = line.reshape(4)
            for x1, y1, x2, y2 in line:
                parameters = np.polyfit((x1, x2), (y1, y2), 1)  # lệnh tính slope và intercept từ 4 điểm cho trước
                slope = parameters[0]
                intercept = parameters[1]
                if slope < 0:   # nếu slope < 0, thì là làn đường bên trái và ngược lại
                    left_fit.append((slope, intercept)) # thêm vào cuối của list
                else:
                    right_fit.append((slope, intercept))

    left_fit_mean = np.mean(left_fit, axis=0)       # lấy trung bình tất cả giá trị của slope và của intercept left_fit
    right_fit_mean = np.mean(right_fit, axis=0)     # lấy trung bình tất cả giá trị của slope và của intercept right_fit
    left_line = create_coordinates(vid, left_fit_mean)
    right_line = create_coordinates(vid, right_fit_mean)
    return np.array([left_line, right_line])

def display_lines(vid, lines):
    line_vid = np.zeros_like(vid) # tạo  một mạng 0 theo kích thước video
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_vid, (x1, y1), (x2, y2), (0, 0, 255), 8)
    return line_vid


while (cap.isOpened()): # kiểm tra khi video mở
    ret, frame = cap.read() # ret là một boolean liên quan đến việc có hay không
    if ret == True:
        canny_vid = canny_edge_detector(frame)
        custom_vid = region(canny_vid)

        lines = cv2.HoughLinesP(custom_vid, 1, np.pi / 180, 100, minLineLength = 115, maxLineGap = 5)

        averaged_lines = average_slope_intercept(frame, lines)
        line_vid = display_lines(frame, averaged_lines)
        result_vid = cv2.addWeighted(frame, 0.9, line_vid, 1, 1) # dùng để kết hợp để cùng hiện video và line vẽ detect làn đường
        cv2.imshow("results", result_vid)
        out.write(result_vid) # xuất video theo định dạnh khai báo từ trước

        if cv2.waitKey(1) & 0xFF == ord('b'):
            break
    else:
        break

# close the video file
cap.release()
out.release()
# destroy all the windows that is currently on
cv2.destroyAllWindows()

