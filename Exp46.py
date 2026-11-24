import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

train_data = tf.keras.utils.image_dataset_from_directory(
    "plant_dataset",
    image_size=(224,224),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_data = tf.keras.utils.image_dataset_from_directory(
    "plant_dataset",
    image_size=(224,224),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=42
)

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Lambda(
        tf.keras.applications.resnet50.preprocess_input
    ),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_data.class_names),
                 activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data,
          validation_data=val_data,
          epochs=5)

loss, accuracy = model.evaluate(val_data)

print("Plant Disease Classification Accuracy:", accuracy)
print("Classes:", train_data.class_names)
