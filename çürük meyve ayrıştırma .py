import numpy as np
import cv2
import time
import serial
import threading

def capture_images():
    global imageFrame
    while True:
        _, imageFrame = webcam.read()


webcam = cv2.VideoCapture(0)


image_capture_thread = threading.Thread(target=capture_images)
image_capture_thread.start()

start_time = time.time()

while True:
    _, imageFrame = webcam.read()
  
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    lower_red = np.array([100, 100, 100])
    upper_red = np.array([180, 255, 255])

    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    mask_green = cv2.inRange(hsvFrame, lower_green, upper_green)
    mask_red = cv2.inRange(hsvFrame, lower_red, upper_red)


    mask_green = cv2.dilate(mask_green, np.ones((5, 5), "uint8"))
    mask_red = cv2.dilate(mask_red, np.ones((5, 5), "uint8"))

    
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    largest_contour = None

    for contour_green in contours_green:
        area = cv2.contourArea(contour_green)
        if area > largest_area:
            largest_area = area
            largest_contour = contour_green

    for contour_red in contours_red:
        area = cv2.contourArea(contour_red)
        if area > largest_area:
            largest_area = area
            largest_contour = contour_red

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_frame = hsvFrame[y:y+h, x:x+w]
        mask_black = cv2.inRange(roi_frame, np.array([0, 0, 0]), np.array([5, 255, 30]))
        mask_black = cv2.dilate(mask_black, np.ones((5, 5), "uint8"))

        contours_black, _ = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_black) > 0:
            x_black, y_black, w_black, h_black = cv2.boundingRect(contours_black[0])
            cv2.rectangle(imageFrame, (x + x_black, y + y_black), (x + x_black + w_black, y + y_black + h_black), (0, 0, 0), 2)

            current_time = time.time()
            if current_time - start_time >= 3:
                data ='1'
                print("merhaba dünya")
           
                print(data)
                start_time = current_time
      
   
    
    cv2.imshow("çürük tespiti", imageFrame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
