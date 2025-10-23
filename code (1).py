import os
import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Constants
IMG_SIZE = (224, 224)  # MobileNetV2 input size
NUM_CLASSES = 60  # As per thesis
BATCH_SIZE = 32
EPOCHS = 50
DATASET_PATH = 'path/to/your/dataset'  # Replace with actual path, e.g., 'dataset/'

# 1. PREPROCESSING MODULE (As per Chapter 3.2)
def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Resize
    img = cv2.resize(img, IMG_SIZE)
    
    # Grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction (Gaussian filter)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Normalization (0-1)
    gray = gray / 255.0
    
    # Edge detection (Canny)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 100, 200)
    
    # Convert back to 3-channel for MobileNetV2 (stack grayscale)
    processed = np.stack([gray, gray, gray], axis=-1)
    
    return processed

# Data augmentation (as per thesis)
def create_data_generators(train_dir, test_dir):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_generator, test_generator

# 2. SEGMENTATION AND FEATURE EXTRACTION MODULE (As per Chapter 3.3 and 3.3.1)
# Approximate ALBMS: Multilevel segmentation with lesion detection
def active_lesion_based_segmentation(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Thresholding for rough segmentation
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations for lesion detection
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Multilevel: Dilate for expansion
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Distance transform for sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Watershed for multilevel segmentation
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Mark boundaries
    
    return image

# DCDC: Distance and Color Directional Clustering
def distance_color_directional_clustering(image, n_clusters=3):
    # Flatten image for clustering
    pixels = image.reshape(-1, 3)
    
    # K-Means clustering (approximates DCDC with color and spatial distance)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    
    # Reshape back
    segmented = labels.reshape(image.shape[:2])
    
    # Refine with morphological operations
    segmented = cv2.morphologyEx(segmented.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    return segmented

# Gabor Filters for texture extraction
def apply_gabor_filters(image, frequencies=[0.1, 0.2], orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    features = []
    for freq in frequencies:
        for theta in orientations:
            kernel = cv2.getGaborKernel((21, 21), 5, theta, freq, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            features.append(filtered.mean())  # Extract mean as feature
    return np.array(features)

# C-Means Clustering for feature grouping
def c_means_clustering(features, n_clusters=5):
    from sklearn.cluster import KMeans  # Approximate C-Means with K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features.reshape(-1, 1))
    return clusters

# Full feature extraction pipeline
def extract_features(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return None
    
    # Segmentation
    segmented = active_lesion_based_segmentation(img)
    clustered = distance_color_directional_clustering(segmented)
    
    # Gabor features
    gabor_features = apply_gabor_filters(img[:, :, 0])  # Use grayscale channel
    
    # C-Means on features
    clustered_features = c_means_clustering(gabor_features)
    
    # Combine features (flatten for input to model)
    combined = np.concatenate([gabor_features, clustered_features])
    return combined

# 3. CLASSIFICATION MODULE (As per Chapter 3.4 and 4.3)
def build_mobilenetv2_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base layers for transfer learning
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training
def train_model(train_generator, test_generator):
    model = build_mobilenetv2_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        callbacks=[early_stop]
    )
    return model, history

# Evaluation
def evaluate_model(model, test_generator):
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Paths (update to your dataset)
    train_dir = os.path.join(DATASET_PATH, 'train')
    test_dir = os.path.join(DATASET_PATH, 'test')
    
    # Create generators
    train_generator, test_generator = create_data_generators(train_dir, test_dir)
    
    # Train model
    model, history = train_model(train_generator, test_generator)
    
    # Evaluate
    evaluate_model(model, test_generator)
    
    # Save model
    model.save('medicinal_plant_classifier.h5')
    
    # Example prediction
    sample_image_path = 'path/to/sample/image.jpg'
    features = extract_features(sample_image_path)
    if features is not None:
        # For prediction, you'd need to preprocess and pass through model
        print("Features extracted successfully.")
