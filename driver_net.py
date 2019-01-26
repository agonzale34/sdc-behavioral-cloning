import csv
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D

lines = []
angle_correction = 0.2
data_folder = './training-data/'


def add_image_list(img_list, in_img_path):
    img_name_path = data_folder + 'IMG/' + in_img_path.split('/')[-1]
    img_list.append(plt.imread(img_name_path))


# read all the lines in the csv file
with open(data_folder + 'driving_main_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

# load the images
images = []
angles = []

print('Loading', len(lines),  'images... please wait')
for line in lines:
    angle_center = float(line[3])
    angle_left = angle_center + angle_correction
    angle_right = angle_center - angle_correction
    angles = angles + [angle_center, angle_left, angle_right]
    add_image_list(images, line[0])
    add_image_list(images, line[1])
    add_image_list(images, line[2])

# creating the dataset for training the model
X_train = np.array(images)
y_train = np.array(angles)
print('Dataset loaded successfully with shape:', images[0].shape)
print('Number of features:', len(X_train))

RELU = 'relu'
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation=RELU))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation=RELU))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation=RELU))
model.add(Conv2D(64, kernel_size=(3, 3), activation=RELU))
model.add(Conv2D(64, kernel_size=(3, 3), activation=RELU))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
print('DriverNet created successfully')

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)

# Save the model
model.save('model.h5')
print('DriverNet trained and saved successfully')
