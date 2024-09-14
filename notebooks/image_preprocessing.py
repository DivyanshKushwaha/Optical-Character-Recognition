import os
from pathlib import Path
import cv2
import pytesseract
import numpy 
from PIL import Image
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd =r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'



def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"image does not exist at path: {image_path}")
        return None,None
    else:
        image = cv2.imread(image_path)
        if image is None:
            print(f"failed to load the image")
            return None,None
        try:
            gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            print(f"Error during image conversion: {e}")
            return None,None
        resized_image = cv2.resize(gray_image, (224,224))
        text = pytesseract.image_to_string(resized_image)
        return resized_image,text
    
def extract_text(image):
    text = pytesseract.image_to_string(image)
    return text