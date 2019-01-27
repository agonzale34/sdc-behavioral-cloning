import csv

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, BatchNormalization, Activation
from keras.models import Sequential

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

print('Loading images... please wait')
# for line in lines:
for i in range(0, len(lines), 2):
    angle_center = float(lines[i][3])
    angle_left = angle_center + angle_correction
    angle_right = angle_center - angle_correction
    angles = angles + [angle_center, angle_left, angle_right]
    add_image_list(images, lines[i][0])
    add_image_list(images, lines[i][1])
    add_image_list(images, lines[i][2])
    # data augmentation
    images.append(np.fliplr(plt.imread(lines[i][0])))
    images.append(np.fliplr(plt.imread(lines[i][1])))
    images.append(np.fliplr(plt.imread(lines[i][2])))
    angles.append(-angle_center)
    angles.append(-angle_left)
    angles.append(-angle_right)

# creating the dataset for training the model
X_train = np.array(images)
y_train = np.array(angles)
print('Dataset loaded successfully -> features:', len(angles))

RELU = 'relu'
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation=RELU))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation=RELU))
# model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation=RELU))
model.add(Conv2D(64, kernel_size=(3, 3), activation=RELU))
# model.add(Conv2D(64, kernel_size=(3, 3), activation=RELU))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
print('DriverNet created successfully')

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=7)

# Save the model
model.save('model.h5')
print('DriverNet trained and saved successfully')

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()