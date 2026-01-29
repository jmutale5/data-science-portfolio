import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential, layers
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================
# STEP 1. Config & Data Loading
# ===================================
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
DATA_DIR = 'hair_type_data'

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
num_classes = len(class_names)

# ===================================
# STEP 2. Model architecture
# ===================================
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                         include_top=False, weights='imagenet')
base_model.trainable = False

data_augmentation = Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
], name="data_augmentation")

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

# ===========================================
# 3. Stage 1: Transfer Learning (Frozen Base)
# ===========================================
print("\n--- Starting Stage 1: Training Classification Head ---")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping_stage1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stopping_stage1])

# ==============================================
# 4. Stage 2: Fine-Tuning (Unfreezing Top Layers)
# ==============================================
print("\n--- Starting Stage 2: Fine-Tuning ---")
base_model.trainable = True
for layer in base_model.layers[:-100]: # Freeze the first layers
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping_fine_tune = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stopping_fine_tune])

# ===================================
# 5. Model Evaluation
# ===================================
print("\n--- Final Model Evaluation ---")
y_true = []
y_pred = []

# Validation dataset
for images, labels in val_ds:
    y_true.extend(labels.numpy())
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

# Generating Metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Saving the model
model.save('best_hair_model.keras')
print("Model saved as 'best_hair_model.keras'")

# ===================================
# 6. Plot Confusion Matrix
# ===================================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Final Confusion Matrix: Hair Type Classification')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()