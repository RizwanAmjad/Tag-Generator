import numpy as np
import pandas as pd

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

labels_df = pd.read_csv('dataset/train.csv')
images_set = []

no_of_batches = 30
labels_df = labels_df.sample(frac=1)
labels_df['batch'] = np.random.randint(0, no_of_batches, size=len(labels_df))
print(labels_df.tail())
labels_df.groupby(by='batch')

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
model.add(Dense(19, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

for batch in range(0, no_of_batches):
    data_batch = labels_df[labels_df['batch'] == batch]

    for i in data_batch.iterrows():
        image_temp = image.load_img(
            'dataset/Images/' + i[1].Category[2:-2] + '/' + str(i[1].Id) + '.jpg',
            target_size=(300, 300, 3))
        # print('dataset/Images/'+i+'.jpg')
        image_temp = image.img_to_array(image_temp)
        image_temp = image_temp / 255
        images_set.append(image_temp)

    images_set = np.array(images_set)

    labels_set = []

    # Now are going to drop useless columns and then convert the result into a numpy array
    labels_set = np.array(data_batch.drop(['Id', 'Category'], axis=1))

    # Now we are going to create aur keras model
    # I have Copied it from analytics vidhiya website
    print(len(images_set))
    print(len(labels_set))
    training_images, testing_images, training_labels, testing_labels = train_test_split(images_set, labels_set,
                                                                                        random_state=42, test_size=0.1)
    # Delete useless variables to free memory

    model.fit(training_images, training_labels, validation_data=(testing_images, testing_labels), epochs=40)

    del images_set
    del labels_set

# model.fit_generator(training_data, validation_data=None, epochs=30)
# model.fit(images_set, labels_set, epochs=50)

# This below line is not yet tested!
model.save('tags_model.h5')
