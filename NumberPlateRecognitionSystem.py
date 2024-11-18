import cv2
import pickle
import logging
import uvicorn
import warnings
from ultralytics import YOLO
import numpy as np
from paddleocr import PaddleOCR
from fastapi import FastAPI, UploadFile, File, HTTPException
# Suppress specific deprecation warnings
from cryptography.utils import CryptographyDeprecationWarning


# Reduce log level for PaddleOCR to avoid verbose output
logging.getLogger('ppocr').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

app = FastAPI()
class car_detection:
    def __init__(self):
        self.model=YOLO("yolov10l.pt")


    def detect_and_process_cars(self,img):
        # Load YOLO model
        
        # Read and convert image
        img = img
        
        # Get detection results
        result = self.model(img)
        
        # Store cropped images
        car_crops = []
        processed_results = []
        
        # Process each detection
        for box in result[0].boxes:
            # Check if detection is a car (assuming class 2 is car)
            if box.cls == 2:
                # Get confidence
                conf = float(box.conf)
                
                if conf >= 0.55:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]  # Convert tensor to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Crop the car image
                    car_crop = img[y1:y2, x1:x2]
                    
                    # Store cropped image
                    car_crops.append(car_crop)
        return car_crops


       
class LicensePlateDetector:
    def __init__(self):
        """
        Initialize the License Plate Detector
        """
        self.model = None
        
    def initialize_model(self):
        """
        Initialize the model using Roboflow API
        """
        # Load (deserialize) the model from the file
        with open("license_plate_model.pkl", "rb") as file:
            loaded_model = pickle.load(file)
            print("Model has been loaded from 'license_plate_model.pkl'")
        self.model = loaded_model
        # print("Model initialized successfully")
        
    def detect_license_plate(self, image, confidence=0, overlap=40):
        """
        Detect license plates in an image
        
        Parameters:
        image_path (str): Path to the input image
        confidence (int): Confidence threshold (0-100)
        overlap (int): Overlap threshold (0-100)
        
        Returns:
        tuple: (processed image with detections, list of detected plates)
        """
        if self.model is None:
            raise Exception("Model not initialized! Please initialize the model first.")
            
        
        # Get predictions
        predictions = self.model.predict(image,confidence=confidence, overlap=overlap).json()
  
        # Draw boxes around detected license plates
        for prediction in predictions['predictions']:
            # Get coordinates
            x = prediction['x']
            y = prediction['y']
            width = prediction['width']
            height = prediction['height']
            # confidence = prediction['confidence']
            
            # Calculate box coordinates
            x1 = int(x - width/2)
            y1 = int(y - height/2)
            x2 = int(x + width/2)
            y2 = int(y + height/2)
            
            # Ensure coordinates are within image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            
            # Extract the license plate region
            if y2 > y1 and x2 > x1:  # Check if coordinates are valid
                 return image[y1:y2, x1:x2]
            
class ImageTextDetector:
    def __init__(self):
        self.ocr = PaddleOCR(lang='en') # need to run only once to load model into memory

    def detector(self,img):  
        result = self.ocr.ocr(img, det=False, cls=False)
        return result


CD=car_detection()
NPR = LicensePlateDetector()  # Create an instance of NumberPlateRecognition
NPR.initialize_model()      
ITD=ImageTextDetector()

def manager(image):

    images=CD.detect_and_process_cars(image)
    print(len(images))
    result=[]
    for image in images:
        Recognitize_image=NPR.detect_license_plate(image)  # Run number recognition

        if Recognitize_image is None:
             continue
        else:
            result.append(ITD.detector(Recognitize_image))

    return result


# import cv2
# import numpy as np

app = FastAPI()


@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No image part")

    # Read the image file as a numpy array
    image_data = await file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Run the manager function and get the text
    result_text = manager(image)

    return {"Plates": result_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



# image_path='/home/aman/a896/new_project/car numberplet detacetion/image/1.jpeg'
# image=cv2.imread(image_path)
# print(manager(image))