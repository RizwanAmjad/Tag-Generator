import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image


class TagGenerator:
    def __init__(self):
        # importing the Model that generates tags
        self.model = models.load_model('tags_model.h5')

    def get_numpy_array(img_path):
        """
        takes image path as a parameter
        :return: returns numpy array for that image
        """
        try:
            image_loaded = image.load_img(img_path, target_size=(300, 300))
        except Exception:
            raise FileNotFoundError

        image_array = image.img_to_array(image_loaded)
        image_array = image_array / 255
        return np.expand_dims(image_array, axis=0)

    def generate(self, img_path):
        """
        Loads image and generate an array telling about the tags detected in the image/movie cover
        :param img_path: it path of the image
        :return: array with probability of each tag in the image

        """

        # print(image_array)
        image_array = TagGenerator.get_numpy_array(img_path)
        return self.model.predict(image_array)[0]



