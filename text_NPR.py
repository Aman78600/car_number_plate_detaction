import requests
import cv2
import sys
import mimetypes

# URL for the FastAPI endpoint
link = 'http://0.0.0.0:8000/process_image'

# Get the image path from the user input
image_path = input("Enter the path to the image: ")

# Read the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read the image file.")
    sys.exit()

# Determine the file extension and MIME type
file_extension = image_path.split('.')[-1]
mime_type = mimetypes.guess_type(image_path)[0] or 'application/octet-stream'

# Encode image as a byte array
_, image_encoded = cv2.imencode(f'.{file_extension}', image)
image_bytes = image_encoded.tobytes()

# Send POST request with the image file
files = {'file': (f'image.{file_extension}', image_bytes, mime_type)}
response = requests.post(link, files=files)

# Check if the request was successful
if response.status_code == 200:
    print(response.json())
else:
    print('Request failed with status code:', response.status_code, response.text)
