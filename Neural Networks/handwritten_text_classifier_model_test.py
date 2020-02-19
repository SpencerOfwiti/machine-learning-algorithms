# pull scanned image data from github using the terminal
# !git clone https://github.com/crash-course-ai/lab1-neural-networks.git
# !git pull
# !ls lab1-neural-networks/letters_mod
# !cd /content/lab1-neural-networks/letters_mod
# !pwd

# importing libraries
import os
import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np

# loading the model
pickle_in = open('./handwritten_text_classifier_model.pickle', 'rb')
model = pickle.load(pickle_in)

# puts all scanned image data in the 'files' variable
path, dirs, files = next(os.walk('../lab1-neural-networks/letters_mod/'))
files.sort()

# importing the letters, resizing them and printing them out
# process all the scanned images and adds them to the handwritten_story
handwritten_story = []
for i in range(len(files)):
    img = cv2.imread('../lab1-neural-networks/letters_mod/'+files[i], cv2.IMREAD_GRAYSCALE)
    handwritten_story.append(img)

# processing the handwritten story to 28 by 28 pixels
processed_story = []

for img in handwritten_story:
    # Apply Gaussian blur filter
    img = cv2.GaussianBlur(img, (7, 7), 0)

    # extract the region of interest in the image and center in square
    points = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(points)
    if w > 0 and h > 0:
        if w > h:
            y = y - (w - h) // 2
            img = img[y: y+w, x: x+w]
        else:
            x = x - (h - w) // 2
            img = img[y: y+h, x: x+h]

    # resize and resample to be 28 x 28 pixels
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)

    # Normalize the pixels and reshape before adding to the new story array
    img = img / 255.0
    img = img.reshape((28, 28))
    processed_story.append(img)

# displaying an individual letter of the story
plt.imshow(processed_story[4])
plt.show()

# feeding the handwritten story into the model
typed_story = ''
for letter in processed_story:
    letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_CUBIC)

    # checking for blank images
    total_pixel_value = 0
    for j in range(28):
        for k in range(28):
            total_pixel_value += letter[j, k]

    # if less than 20 pixels filled add a space
    if total_pixel_value < 20:
        typed_story = typed_story + ' '
    else:
        single_item_array = (np.array(letter)).reshape(1, 784)
        prediction = model.predict(single_item_array)
        typed_story = typed_story + str(chr(prediction[0]+96))

print(typed_story)

