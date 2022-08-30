import cv2,win32clipboard,os,sys
import numpy as np
from io import BytesIO
from PIL import Image
from tkinter import filedialog

def send_to_clipboard(image_path):
    output = BytesIO()
    Image.open(image_path).convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    win32clipboard.CloseClipboard()

def blur(image):
    model = cv2.dnn.readNetFromCaffe("face_blur\prototxt.txt", "face_blur\model.caffemodel")
    h, w = image.shape[:2]
    model.setInput(cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0)))
    output = np.squeeze(model.forward())
    for i in range(output.shape[0]):
        if output[i, 2] > 0.2:
            start_x, start_y, end_x, end_y = (output[i, 3:7] * np.array([w, h, w, h])).astype(np.int64)
            image[start_y: end_y, start_x: end_x] = cv2.GaussianBlur(image[start_y: end_y, start_x: end_x], ((w // 7) | 1, (h // 7) | 1), 0)
    

def blur_face(image_path, copy_to_clipboard=False,save_image=False,show_image=False):
    image = cv2.imread(image_path)
    blur(image)
    if copy_to_clipboard:
        cv2.imwrite("image_blurred.jpg", image)
        send_to_clipboard('image_blurred.jpg')
        os.remove("image_blurred.jpg")
        print("Image copied to clipboard")
    if save_image:
        cv2.imwrite("image_blurred.jpg", image)
        print("Image saved")
    if show_image:
        cv2.imshow("image", image)
        cv2.waitKey(0)

def live_blur():
    cap = cv2.VideoCapture(0)
    while 1:
        _, image = cap.read()
        blur(image)
        cv2.imshow("image", image)
        if cv2.waitKey(1) > -1:
            break

def blur_face_video(video_path):
    cap = cv2.VideoCapture(video_path)
    _, image = cap.read()
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (image.shape[1], image.shape[0]))
    while 1:
        captured, image = cap.read()
        if not captured:
            break
        blur(image)
        cv2.imshow("image", image)
        if cv2.waitKey(1) > -1:
            break
        out.write(image)

    cv2.destroyAllWindows()
    cap.release()
    out.release()
    
blur_face(filedialog.askopenfilenames()[0], copy_to_clipboard=True)
# live_blur()
# blur_face_video(filedialog.askopenfilenames()[0])