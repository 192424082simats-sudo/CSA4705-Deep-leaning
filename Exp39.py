
import tensorflow as tf
import matplotlib.pyplot as plt


(X_train, _), _ = tf.keras.datasets.mnist.load_data()
image = X_train[0].astype("float32") / 255.0
image = image.reshape(1, 28, 28, 1)


kernel = tf.constant([
    [[[-1.0]], [[0.0]], [[1.0]]],
    [[[-1.0]], [[0.0]], [[1.0]]],
    [[[-1.0]], [[0.0]], [[1.0]]]
])


feature_map = tf.nn.conv2d(
    image, kernel, strides=1, padding='SAME'
)


feature_map = tf.nn.relu(feature_map)


pooled = tf.nn.max_pool2d(
    feature_map, ksize=2, strides=2, padding='VALID'
)

plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.imshow(image[0, :, :, 0], cmap='gray')
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(feature_map[0, :, :, 0], cmap='gray')
plt.title("Convolution")

plt.subplot(1, 3, 3)
plt.imshow(pooled[0, :, :, 0], cmap='gray')
plt.title("Max Pooling")

plt.show()
