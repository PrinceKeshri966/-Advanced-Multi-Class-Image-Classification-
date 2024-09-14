import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from scipy.spatial.distance import cosine

# Load the pre-trained VGG16 model (excluding the top fully connected layers)
model = VGG16(weights='imagenet', include_top=False)

# Preprocess the image to fit the model's input requirements
def preprocess_image(image_path):
    # Read and resize the image to the required input size (224x224)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # VGG16 expects 224x224 images
    
    # Convert the image from BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess the image using the specific VGG16 preprocessing function
    img = preprocess_input(img)
    
    # Expand dimensions to fit the model's expected batch input shape
    img = np.expand_dims(img, axis=0)
    return img

# Compare two images by extracting their features and calculating their similarity
def compare_images(image1_path, image2_path, model):
    # Preprocess both images
    img1 = preprocess_image(image1_path)
    img2 = preprocess_image(image2_path)

    # Extract feature maps for both images using the VGG16 model
    features1 = model.predict(img1).flatten()  # Flatten to 1D
    features2 = model.predict(img2).flatten()  # Flatten to 1D

    # Calculate similarity using cosine distance (smaller value means more similarity)
    similarity = cosine(features1, features2)

    # Set a threshold to determine whether the images are similar
    threshold = 0.5  # 0 = exact match, 1 = complete opposite
    return similarity < threshold

# Paths to the images you want to compare
image1_path = "C:\\Users\\hp\\Desktop\\1.jpg"
image2_path = "C:\\Users\\hp\\Desktop\\5.jpg"

# Perform the image comparison and print the result
match = compare_images(image1_path, image2_path, model)
if match:
    print("The images match the same breed.")
else:
    print("The images do not match the same breed.")
