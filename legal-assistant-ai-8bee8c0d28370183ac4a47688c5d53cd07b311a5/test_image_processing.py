import sys
import torch
import torchvision
import easyocr
import cv2
import numpy as np
from PIL import Image

print("Testing image processing dependencies...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"EasyOCR version: {easyocr.__version__}")

# Test PyTorch
print("\nTesting PyTorch...")
try:
    x = torch.rand(5, 3)
    print(f"PyTorch is working. Random tensor: {x}")
except Exception as e:
    print(f"PyTorch error: {e}")

# Test OpenCV
print("\nTesting OpenCV...")
try:
    # Create a blank image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (90, 90), (0, 255, 0), 2)
    cv2.putText(img, "Test", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    print("OpenCV is working. Created a test image.")
except Exception as e:
    print(f"OpenCV error: {e}")

# Test EasyOCR
print("\nTesting EasyOCR...")
try:
    # This will download the model if not already present
    reader = easyocr.Reader(['en'])
    print("EasyOCR initialized successfully.")
except Exception as e:
    print(f"EasyOCR error: {e}")

print("\nAll tests completed.")
