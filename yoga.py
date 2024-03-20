from tensorflow.keras.models import load_model
import cv2
import numpy as np
from keras.models import load_model
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk
from datetime import datetime
import time
# Load the saved model
model = load_model('C:/Users/user/Downloads/final_model.h5')
# Define the camera object
cap = cv2.VideoCapture(0)

# Define the GUI window
root = Tk()
root.title("Yoga Posture Classifier")

# Define the label to display the predicted posture
posture_label = Label(root, text="", font=("Arial", 30))
posture_label.pack(pady=20)



def predict_posture(start_time,x):
    # Capture the image
    ret, bgr_img = cap.read()
    label_position = (bgr_img.shape[1]-150, bgr_img.shape[0]-50) # bottom right corner
    cv2.putText(bgr_img, x, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('image', bgr_img)
    # Preprocess the image
    # dsize
    start_time1 = time.time()
    if ((((int)(start_time1)-(int)(start_time)) % 5 == 0) ):
        print(start_time1+1)
        #print(x)
        #x=start_time1
        dsize = (64,64)
        #resize image
        resized_image = cv2.resize(bgr_img,dsize)
        # convert from BGR color-space to YCrCb
        ycrcb_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YCrCb)
        # create a CLAHE object 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # Now apply CLAHE object on the YCrCb image
        ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
        # convertback to BGR color-space from YCrCb
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        # Denoise is done to remove unwanted noise to better perform
        equalized_denoised_image = cv2.fastNlMeansDenoisingColored(equalized_img, None, 10, 10, 7, 21)
        image=np.array([equalized_denoised_image/255])
        pred=model.predict(image)
        # Predict the posture
        pred=np.squeeze(np.where(pred == np.max(np.squeeze(pred)), 1, 0))
        if pred[0]==1:
            posture_name="downdog"
        elif pred[1]==1:
            posture_name="tree"
        elif pred[2]==1:
            posture_name="plank"
        elif pred[3]==1:
            posture_name="warrior2"
        else:
            posture_name="goddess"
        # Display the predicted posture name
        #posture_label.config(text=posture_name)
        print(posture_name)
        # Add label to the image


        return posture_name
    if x == "":
        return ""
    return x


# Run the GUI window
# Call the function to start predicting postures
start_time = time.time()
x=""
while True:
    x=predict_posture(start_time,x)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
