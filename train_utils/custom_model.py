import tensorflow as tf

from train_utils.augmenters import data_augmenter, data_augmenter_fruit360
from train_utils.utils import is_in


class CustomModel(tf.keras.Model):
    def __init__(self, all_classes=None, fruit_classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.food_augmenter = data_augmenter()
        self.fruit_augmenter = data_augmenter_fruit360()
        if all_classes is None or fruit_classes is None:
            self.fruit_indices = tf.constant(
                []
            )
        else:
            self.fruit_indices = tf.constant(
                [all_classes.index(fruit) for fruit in fruit_classes]
            )

    def train_step(self, data):
        x, y = data
        class_indices = tf.math.argmax(y, -1)
        isfr = is_in(class_indices, self.fruit_indices)
        isfr = tf.expand_dims(tf.expand_dims(tf.expand_dims(isfr, -1), -1), -1)
        x = tf.where(
            isfr,
            self.fruit_augmenter(x),
            self.food_augmenter(x),
        )
        return super().train_step((x, y))
