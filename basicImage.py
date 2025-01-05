import cv2
import numpy as np

# Step 1: Read the Image
image_path = 'maze.png'
image = cv2.imread(image_path)

# Step 2: Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Thresholding to get a binary image
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Step 4: Detect the four corners of the maze
# Assuming the maze is rectangular and occupies the entire image
height, width = binary.shape
# corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]

# Step 5: Create the Binary Matrix
# Initialize the matrix with the same dimensions as the binary image
maze_matrix = np.full(binary.shape, 0, dtype=object)


for y in range(height):
    for x in range(width):
        if binary[y, x] == 255:
            maze_matrix[y, x] = 1
        else:
            maze_matrix[y, x] = 0

# Print the resulting matrix
for row in maze_matrix:
    print(' '.join(map(str, row)))