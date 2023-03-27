import cv2
import pytesseract

# Load the image and convert it to grayscale
image = cv2.imread('image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the configuration parameters for text detection
config = ('-l eng --oem 1 --psm 3')

# Use Tesseract to detect the bounding box of the address box
d = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
n_boxes = len(d['text'])
for i in range(n_boxes):
    if d['text'][i].strip().lower() == 'address':
        x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        address_box = gray[y:y+h, x:x+w]

# Extract the text from the address box using pytesseract
config = ('-l eng --oem 1 --psm 6')
address_text = pytesseract.image_to_string(address_box, config=config)

# Print the address text
print(address_text)
