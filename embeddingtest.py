import tensorflow as tf
embedding = tf.keras.layers.Embedding(4, 3)
embedding(tf.convert_to_tensor([1, 2, 3, 1]))
embedding.get_weights()[0]
