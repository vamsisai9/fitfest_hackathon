import pytesseract
import cv2
import pandas as pd

# Read input image
img = cv2.imread('../resource/asnlib/publicdata/invoice.jpg', cv2.IMREAD_GRAYSCALE)

# Preprocess image
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Perform OCR and get data as a table
data = pytesseract.image_to_data(img, output_type='data.frame')

# Filter out non-text regions
data = data[data.conf != -1]
data = data[data.text.str.contains('[a-zA-Z0-9]')]

# Group by line number
lines = data.groupby('block_num')['text'].apply(list).apply(lambda x: ' '.join(x)).reset_index()

# Create dataframe with columns for each cell in the table
table = pd.DataFrame(columns=['Name', 'Quantity', 'Price', 'Total'])

# Extract data for each cell
for i, row in lines.iterrows():
    line_text = row['text'].lower()
    if 'name' in line_text:
        table.at[0, 'Name'] = lines.iloc[i+1]['text']
    elif 'quantity' in line_text:
        table.at[0, 'Quantity'] = lines.iloc[i+1]['text']
    elif 'price' in line_text:
        table.at[0, 'Price'] = lines.iloc[i+1]['text']
    elif 'total' in line_text:
        table.at[0, 'Total'] = lines.iloc[i+1]['text']

# Print table data
print(table)
