import numpy as np
import pandas as pd

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.losses import binary_crossentropy

# training_data_generator = ImageDataGenerator(rescale=1/255)
# training_data = training_data_generator.flow_from_directory(
#
#     "dataset/train", target_size=(300, 300))

# valid_data_generator = ImageDataGenerator(rescale=1/255)
# valid_data = training_data_generator.flow_from_directory(
#     "dataset/valid", target_size=(300, 300))

#We are loading labels and image names from the csv file here
labels_df = pd.read_csv('dataset/train.csv')
images_set = []

for i in labels_df.Id:
    image_temp = image.load_img('dataset/Images/'+i+'.jpg', target_size=(300, 300, 3))
    image_temp = image.img_to_array(image_temp)
    image_temp = image_temp/255
    images_set.append(image_temp)



images_set = np.array(images_set)

labels_set = []

# Now are going to drop useless columns and then convert the result into a numpy array
labels_set = np.array(labels_df.drop(['Id', 'Category'], axis=1))
print(len(labels_set), 'Labels read')

# Now we are going to create aur keras model
# I have Copied it from analytics vidhiya website
print(images_set)
print(labels_set)
# training_images, training_labels, testing_images, testing_labels = train_test_split(images_set, labels_set, random_state=42, test_size=0.1)
# training_images = images_set[:6550]
# testing_images = images_set[6550:]
#
# training_labels = labels_set[:6550]
# testing_labels = labels_set[6550:]

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(300, 300, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='sigmoid'))

# Now we are going to Compile our model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# from tensorflow.keras import models
#
# model = models.load_model('tags_model.h5')

# model.fit_generator(training_data, validation_data=None, epochs=30)
model.fit(images_set, labels_set, epochs=45)
model.save('tags_model.h5')