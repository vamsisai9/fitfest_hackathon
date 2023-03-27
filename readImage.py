import cv2
import pytesseract

# Read image
img = cv2.imread('invoice.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply image processing techniques to extract text
text = pytesseract.image_to_string(gray)

# Print extracted text
print(text)
