# pip install opencv-python
import cv2
# from cv2 import cv2

webcam=cv2.VideoCapture(1)

while True:
    ret,frame=webcam.read()

    if ret==True:
        cv2.imshow("Koolac",frame)
        key=cv2.waitKey(20) & 0xFF
        if key==ord("q"):
            break

webcam.release()
cv2.destroyAllWindows()