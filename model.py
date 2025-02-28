import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory('dataset/', target_size=(224, 224), batch_size=32, subset='training')
val_data = datagen.flow_from_directory('dataset/', target_size=(224, 224), batch_size=32, subset='validation')

model.fit(train_data, validation_data=val_data, epochs=10)

model.save("printhead_detector.h5")