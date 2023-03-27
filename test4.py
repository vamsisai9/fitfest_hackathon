import cv2
import pytesseract

# Read the image and convert it to grayscale
image = cv2.imread('document.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to convert the image to black and white
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Perform connected component analysis to isolate text regions
nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

# Loop over the regions and extract the text
for i in range(1, nLabels):
    # Extract the region and calculate its font characteristics
    x, y, w, h, area = stats[i]
    region = image[y:y+h, x:x+w]
    text = pytesseract.image_to_string(region)
    boldness = calculate_boldness(region)

    # If the region corresponds to a heading, print it
    if boldness > 0.8:
        print('Heading:', text)
