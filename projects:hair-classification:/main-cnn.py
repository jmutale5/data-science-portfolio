# ===================================
# STEP 1: LOADING DATA
# ===================================
import tensorflow as tf

# Defining parameters
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
DATA_DIR = 'hair_type_data'

# Loading & splitting training and validation data (80/20 split)
train_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(f"Found {len(class_names)} curl classes: {class_names}")

# ===================================
# STEP 2: CREATING A SIMPLE CNN MODEL
# ===================================
from tensorflow.keras import layers, Sequential

num_classes = len(class_names)

model = Sequential([
  # Rescaling to 0-1
  layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

  # Convolutional Block 1
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  # Convolutional Block 2
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  # Output Layers
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])

# ===================================
# STEP 3: COMPILATION & TRAINING\
# ===================================

# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Training the model
epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
# ===================================
# IMPROVING MODEL PERFORMANCE
# ===================================

# Summary: Overfitting
    # accuracy (training data) 99.27%
    # val_accuracy (validation data) 62.46%
    # loss 0.0628
    # val_loss 1.3747

from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping  # New Import!

# 1. Defining augmentation layers
data_augmentation = Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
], name="data_augmentation")

num_classes = len(class_names)

# 2. Updating model architecture
model = Sequential([
    # Input Layer is now implicitly handled by the next layer
    layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    data_augmentation,

    # Convolutional Block 1
    layers.Conv2D(32, 3, padding='same', activation='relu'),  # Increased filters
    layers.MaxPooling2D(),

    # Convolutional Block 2
    layers.Conv2D(64, 3, padding='same', activation='relu'),  # Increased filters
    layers.MaxPooling2D(),

    # Output Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # New Dropout layer for regularization
    layers.Dense(num_classes, activation='softmax')
])

# 3. Defining the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Pay attention to this
    patience=5,  # How many epochs to wait before stopping
    restore_best_weights=True
)

# 4. Re-train with callback
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stopping]
)
