import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image, ImageDraw
import numpy as np
import cv2

# load the trained model to classify sign
from keras.models import load_model

model = load_model('traffic_classifier.h5')

# dictionary to label all traffic signs class
classes = {
    1: 'Speed limit (20km/h)',
    2: 'Speed limit (30km/h)',
    3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)',
    5: 'Speed limit (70km/h)',
    6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)',
    8: 'Speed limit (100km/h)',
    9: 'Speed limit (120km/h)',
    10: 'No passing',
    11: 'No passing veh over 3.5 tons',
    12: 'Right-of-way at intersection',
    13: 'Priority road',
    14: 'Yield',
    15: 'Stop',
    16: 'No vehicles',
    17: 'Veh > 3.5 tons prohibited',
    18: 'No entry',
    19: 'General caution',
    20: 'Dangerous curve left',
    21: 'Dangerous curve right',
    22: 'Double curve',
    23: 'Bumpy road',
    24: 'Slippery road',
    25: 'Road narrows on the right',
    26: 'Road work',
    27: 'Traffic signals',
    28: 'Pedestrians',
    29: 'Children crossing',
    30: 'Bicycles crossing',
    31: 'Beware of ice/snow',
    32: 'Wild animals crossing',
    33: 'End speed + passing limits',
    34: 'Turn right ahead',
    35: 'Turn left ahead',
    36: 'Ahead only',
    37: 'Go straight or right',
    38: 'Go straight or left',
    39: 'Keep right',
    40: 'Keep left',
    41: 'Roundabout mandatory',
    42: 'End of no passing',
    43: 'End no passing vehicle with a weight greater than 3.5 tons'
}

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def detect_shapes(img_path):
    img = cv2.imread(img_path)   
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    imgCan = cv2.Canny(imgGray, 255, 255)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCan, kernel, iterations=1)
    
    cropped_shapes = []
    contours, _ = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x_, y_, w_, h_ = cv2.boundingRect(approx)
            cropped_shapes.append([x_, y_, w_ + x_, h_ + y_])
    return cropped_shapes

def crop_image_pil(img_path):
    img = Image.open(img_path)
    cropped_images = []
    cropped_shapes = detect_shapes(img_path)
    
    for points in cropped_shapes:
        cropped_img = img.crop((points[0], points[1], points[2], points[3])) 
        cropped_images.append([cropped_img, points])
    
    return cropped_images

def draw_rectangle(img_path, x, y, w, h):
    uploaded = Image.open(img_path)
    uploaded.thumbnail(((top.winfo_width()), (top.winfo_height())))
    draw = ImageDraw.Draw(uploaded)
    draw.rectangle((x, y, w, h), outline='red', width=2)
    im = ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image = im

# def draw_rec(img_path, recs):
#     uploaded = Image.open(img_path)
#     uploaded.thumbnail((top.winfo_width(), top.winfo_height()))
#     draw = ImageDraw.Draw(uploaded)
#     for x in recs:
#         draw.rectangle((x[0], x[1], x[2], x[3]), outline='red', width=2)
#     im = ImageTk.PhotoImage(uploaded)
#     sign_image.configure(image=im)
#     sign_image.image = im

def classify(file_path):
    global label_packed
    
    cropped_imgs = crop_image_pil(file_path)
    
    # rec = []
    res = []
    for x in cropped_imgs:
        img = x[0]
        img = img.convert('RGB')
        img = img.resize((30, 30))
        img = np.expand_dims(img, axis=0)
        img = np.array(img)
        try: 
            
            pred = model.predict([img])[0]
            pred = [[i, prob] for i,prob in enumerate(pred) if prob > 0.7][0]
            sign = classes[pred[0]+1]
            # rec.append([x[1][0], x[1][1], x[1][2], x[1][3]])
            res.append([sign, pred[1]*100, x[1]])
        except Exception as e: 
            pass
        
    # draw_rec(file_path, rec)

    if len(res) == 0:
        label.configure(foreground='#011638', text='No traffic sign detected')
    else:
        sorted_arr = sorted(res, key=lambda x: x[1], reverse=True)
        draw_rectangle(file_path, sorted_arr[0][2][0], sorted_arr[0][2][1], sorted_arr[0][2][2], sorted_arr[0][2][3])
        label.configure(foreground='#011638', text=sorted_arr[0][0] + ' ' + str(sorted_arr[0][1]) + '%')


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()), (top.winfo_height())))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)

sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text="check traffic sign", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()