# Emotion detector
# Importing libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Data Preprocessing
# 1. Preprocessing the Training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        shear_range=0.3,
        zoom_range=0.3,
        width_shift_range=0.4,
        height_shift_range=0.4,
        horizontal_flip=True,
        fill_mode='nearest'
        )
training_set = train_datagen.flow_from_directory(
        '/home/aicore/fer2013/train',
        color_mode='grayscale',
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
        )

# Data Preprocessing
# 2. Preprocessing Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        '/home/aicore/fer2013/validation',
        color_mode='grayscale',
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
        )

# Building the CNN
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Convolution
# first block
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu', input_shape=[48, 48, 1]))
cnn.add(tf.keras.layers.BatchNormalization())

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu', input_shape=[48, 48, 1]))
cnn.add(tf.keras.layers.BatchNormalization())

# Pooling
# Max pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))

# second Block
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))

# third Block
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))

# fourth Block
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Fully connected layer
cnn.add(tf.keras.layers.Dense(units=64, kernel_initializer='he_normal', activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Dropout(0.5))

cnn.add(tf.keras.layers.Dense(units=64, kernel_initializer='he_normal', activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Dropout(0.5))

# output layer
cnn.add(tf.keras.layers.Dense(units=7, kernel_initializer='he_normal', activation='softmax'))

print(cnn.summary())

# saving the model
checkpoint = tf.keras.callbacks.ModelCheckpoint('Emotion_detection.h5',
                                               monitor='val_loss',
                                               mode='min',
                                               save_best_only=True,
                                               verbose=1)

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=9,
                          verbose=1,
                          restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                          min_delta=0.0001,
                          patience=3,
                          verbose=1,
                          factor=0.2)


callbacks = [earlystop, checkpoint, reduce_lr]

# compiling
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

cnn.fit(x = training_set, validation_data = test_set, callbacks=callbacks, epochs = 25)