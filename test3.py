import cv2
import pytesseract

# Load the image
img = cv2.imread('invoice.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu thresholding to binarize the image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Perform OCR on the thresholded image
data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

# Search for the text "billing address" in the recognized words
for i, word in enumerate(data['text']):
    if word.lower() == 'billing' and data['text'][i+1].lower() == 'address':
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i] + data['width'][i+1]
        h = max(data['height'][i], data['height'][i+1])
        print('Billing address box coordinates: x={}, y={}, w={}, h={}'.format(x, y, w, h))
