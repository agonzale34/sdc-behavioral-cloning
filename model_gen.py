import csv

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, BatchNormalization, Activation, \
    MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

angle_correction = 0.2
data_folder = './training-data/'

car_images = []
car_angles = []

# read all the lines in the csv file
with open(data_folder + 'driving_main_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        angle_center = float(line[3])
        angle_left = angle_center + angle_correction
        angle_right = angle_center - angle_correction
        car_images = car_images + [line[0], line[1], line[2]]
        car_angles = car_angles + [angle_center, angle_left, angle_right]

print('Dataset CSV loaded successfully length:', len(car_images))

# creating the train a validation data
train_images, valid_images = train_test_split(car_images, test_size=0.2)
train_angles, valid_angles = train_test_split(car_angles, test_size=0.2)


# define the generator
def generator(in_images, in_angles, batch_size=32):
    n_lines = len(in_images)
    # Loop forever so the generator never terminates
    while 1:
        for offset in range(0, n_lines, batch_size):
            batch_images = in_images[offset:offset + batch_size]
            batch_angles = in_angles[offset:offset + batch_size]

            images = []
            for batch_image in batch_images:
                img_name_path = data_folder + 'IMG/' + batch_image.split('/')[-1]
                images.append(plt.imread(img_name_path))

            X_train = np.array(images)
            y_train = np.array(batch_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


print('Dataset loaded successfully')
print('Number of features:', len(car_images))

RELU = 'relu'
SAME = 'same'
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding=SAME, activation=RELU))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), padding=SAME, activation=RELU))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
# model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), padding=SAME, activation=RELU))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), padding=SAME, activation=RELU))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), padding=SAME, activation=RELU))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
print('DriverNet created successfully')

model.compile(loss='mse', optimizer=Adam(0.0001))

# create the generators
train_generator = generator(train_images, train_angles, batch_size=64)
valid_generator = generator(valid_images, valid_angles, batch_size=64)

history_object = model.fit_generator(
    train_generator, steps_per_epoch=len(train_angles), epochs=5,
    validation_data=valid_generator, validation_steps=len(valid_angles),
    max_queue_size=200, workers=4, use_multiprocessing=True
)

# Save the model
model.save('model.h5')
print('DriverNet trained and saved successfully')

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
