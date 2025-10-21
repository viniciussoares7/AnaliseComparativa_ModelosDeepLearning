import tensorflow as tf
print("Versão do TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Físicas Encontradas:", gpus)