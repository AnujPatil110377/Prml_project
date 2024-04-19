import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
loaded_model = tf.keras.models.load_model("pretrained_ann_model.h5")