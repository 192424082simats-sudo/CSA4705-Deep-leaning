
import tensorflow as tf
import matplotlib.pyplot as plt


image = tf.random.uniform((1, 8, 8, 1))


max_pool = tf.nn.max_pool2d(
    image, ksize=2, strides=2, padding='VALID'
)


avg_pool = tf.nn.avg_pool2d(
    image, ksize=2, strides=2, padding='VALID'
)

plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.imshow(image[0, :, :, 0], cmap='gray')
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(max_pool[0, :, :, 0], cmap='gray')
plt.title("Max Pooling")

plt.subplot(1, 3, 3)
plt.imshow(avg_pool[0, :, :, 0], cmap='gray')
plt.title("Average Pooling")

plt.show()

print("Original shape:", image.shape)
print("Max pooling shape:", max_pool.shape)
print("Average pooling shape:", avg_pool.shape)
