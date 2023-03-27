import cv2
import pytesseract

# Read the image and convert it to grayscale
image = cv2.imread('document.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to convert the image to black and white
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Perform connected component analysis to isolate text regions
nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
def calculate_boldness(region):
    # Convert the region to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Compute the Scharr gradient magnitude representation of the images in both the x and y direction
    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # Compute the magnitude of the gradient
    mag = cv2.magnitude(grad_x, grad_y)

    # Compute the mean intensity of the image
    mean_intensity = np.mean(mag)

    # Compute the standard deviation of the image intensity
    std_intensity = np.std(mag)

    # Compute the ratio of standard deviation to mean intensity as the boldness
    boldness = std_intensity / mean_intensity

    return boldness


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
