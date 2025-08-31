#!/usr/bin/env python3
"""
Pancreatic Cancer Early Detection - Multi-modal ML Analysis
This script combines a tabular data pipeline with a deep learning pipeline for
CT scan images to create a comprehensive, multi-modal detection system.
It uses TensorFlow/Keras for the deep learning component and scikit-learn for
the tabular data.
"""

import pandas as pd
import numpy as np
import warnings
import os
import cv2
from tqdm import tqdm
import zipfile
import shutil
import matplotlib.pyplot as plt
import nibabel as nib # New library for .nii.gz files

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

class PancreaticCancerML:
    """Comprehensive multi-modal ML pipeline for cancer detection analysis"""
    
    def __init__(self, tabular_data_path=None, image_dir_path=None):
        self.tabular_data_path = tabular_data_path
        self.image_dir_path = image_dir_path
        self.tabular_features = None
        self.image_data = None
        self.labels = None
        self.X_train_tab = None
        self.X_test_tab = None
        self.X_train_img = None
        self.X_test_img = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.history = None
        
        # Define image parameters
        self.IMG_SIZE = (128, 128)
        self.BATCH_SIZE = 32
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess both tabular and image data.
        
        NOTE: For this demonstration, we create synthetic tabular data.
        You should replace this part with your actual data loading.
        """
        print("Loading and preprocessing data...")
        
        # --- Step 1: Load and preprocess TABULAR data ---
        print("Creating synthetic tabular data...")
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        X_tab = np.random.randn(n_samples, n_features)
        y = (X_tab[:, 0] + X_tab[:, 1]*2 + X_tab[:, 2]*3 + np.random.randn(n_samples) > 0).astype(int)
        
        self.tabular_features = pd.DataFrame(X_tab, columns=[f'feature_{i}' for i in range(n_features)])
        self.labels = pd.Series(y, name='target')
        
        print(f"Tabular data shape: {self.tabular_features.shape}")
        
        # --- Step 2: Load and preprocess IMAGE data ---
        print("Loading image data from directory...")

        # Zip file names for both the labels (segmentation masks) and the raw CT scans
        zip_file_name_labels = 'TCIA_pancreas_labels-02-05-2017-1.zip'
        extracted_dir_labels = 'TCIA_pancreas_labels-02-05-2017-1'
        zip_file_name_ct = 'TCIA_pancreas_CTs-02-05-2017.zip' # This zip file contains the raw scans
        extracted_dir_ct = 'TCIA_pancreas_CTs-02-05-2017'
        
        self.image_dir_path = extracted_dir_labels

        if not os.path.exists(extracted_dir_labels):
            if os.path.exists(zip_file_name_labels):
                print(f"Unzipping '{zip_file_name_labels}'...")
                with zipfile.ZipFile(zip_file_name_labels, 'r') as zip_ref:
                    zip_ref.extractall(extracted_dir_labels)
                print(f"Extraction complete. Data is in '{extracted_dir_labels}' directory.")
            else:
                print(f"Error: Label zip file '{zip_file_name_labels}' not found.")
                print("Please ensure the zip file is in the same directory as the script.")
                self.image_data = np.array([])
                return
        
        # Check for and unzip the raw CT scan data as well
        if not os.path.exists(extracted_dir_ct):
             if os.path.exists(zip_file_name_ct):
                print(f"Unzipping '{zip_file_name_ct}'...")
                with zipfile.ZipFile(zip_file_name_ct, 'r') as zip_ref:
                    zip_ref.extractall(extracted_dir_ct)
                print(f"Extraction complete. Data is in '{extracted_dir_ct}' directory.")
             else:
                print(f"Warning: Raw CT scan zip file '{zip_file_name_ct}' not found.")
                print("Proceeding with only the labels (segmentation masks).")
        
        image_paths = []
        
        # Walk through the extracted directory to find all image files
        # This is more robust than assuming a specific folder structure.
        print("Searching for images in the extracted directory structure...")
        
        # Prioritize loading the raw CT scans if available
        ct_paths = []
        for root, dirs, files in os.walk(extracted_dir_ct):
            for file in files:
                if file.lower().endswith('.nii.gz') and 'volume' in file.lower():
                    ct_paths.append(os.path.join(root, file))
        if ct_paths:
            image_paths = ct_paths
            print(f"Found {len(image_paths)} raw CT scans. Processing these for a better visual output.")
        else:
            # Fallback to labels if CT scans are not found
            for root, dirs, files in os.walk(extracted_dir_labels):
                for file in files:
                    if file.lower().endswith('.nii.gz') and 'label' in file.lower():
                        image_paths.append(os.path.join(root, file))
            print(f"No raw CT scans found. Found {len(image_paths)} labels (segmentation masks). Processing these.")
        
        
        if not image_paths:
            print("No image files were found in the directory tree.")
            print("Please ensure your zip file contains at least one image with a supported extension.")
            self.image_data = np.array([])
            return
            
        print(f"Found {len(image_paths)} images. Processing...")
        
        processed_images = []
        
        # We'll use a subset of the images to match the synthetic tabular data size
        image_paths_subset = image_paths[:n_samples]
        
        for img_path in tqdm(image_paths_subset):
            try:
                # Handle .nii.gz files separately using nibabel
                if img_path.lower().endswith('.nii.gz'):
                    img_nib = nib.load(img_path)
                    img_data = img_nib.get_fdata()
                    # We need to select a 2D slice from the 3D volume
                    # Here we take the middle slice from the third dimension
                    img = img_data[:, :, img_data.shape[2] // 2]
                else:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize and normalize the image
                    img = cv2.resize(img, self.IMG_SIZE)
                    # Add a channel dimension for Keras
                    img = np.expand_dims(img, axis=-1)
                    img = img / 255.0  # Normalize pixel values
                    processed_images.append(img)
            except Exception as e:
                print(f"Could not process image {img_path}: {e}")
        
        if not processed_images:
            print("No images were processed correctly.")
            self.image_data = np.array([])
            return

        self.image_data = np.array(processed_images)
        
        # Generate synthetic labels for the images to align with the synthetic tabular data.
        self.labels = np.random.randint(2, size=len(self.image_data))
        
        print(f"\nImage data shape: {self.image_data.shape}")
        print(f"Number of loaded images: {len(self.image_data)}")
        
        # Align image data with tabular data
        if len(self.image_data) != n_samples:
            print("Warning: Mismatch between number of tabular samples and images.")
            print("Aligning datasets by using only the first N samples from both.")
            min_samples = min(len(self.image_data), len(self.tabular_features))
            self.tabular_features = self.tabular_features.iloc[:min_samples]
            self.labels = self.labels[:min_samples]
            self.image_data = self.image_data[:min_samples]
            
        print(f"Final aligned dataset size: {len(self.tabular_features)}")

    def split_data(self, test_size=0.2, random_state=42):
        """Split both tabular and image data into training and testing sets"""
        if self.image_data.size == 0 or self.tabular_features.empty:
            print("Data not loaded. Skipping data split.")
            return

        print("Splitting data into training and testing sets...")
        
        # Use a single split for both data types to maintain correspondence
        indices = np.arange(len(self.labels))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=self.labels
        )
        
        self.X_train_tab = self.tabular_features.iloc[train_indices]
        self.X_test_tab = self.tabular_features.iloc[test_indices]
        self.X_train_img = self.image_data[train_indices]
        self.X_test_img = self.image_data[test_indices]
        self.y_train = self.labels[train_indices]
        self.y_test = self.labels[test_indices]
        
        print(f"Training set: Tabular {self.X_train_tab.shape}, Images {self.X_train_img.shape}")
        print(f"Testing set: Tabular {self.X_test_tab.shape}, Images {self.X_test_img.shape}")
    
    def build_multimodal_model(self):
        """Build the combined deep learning model"""
        print("\nBuilding multi-modal model...")
        
        # Input for the tabular data
        tabular_input = keras.Input(shape=(self.tabular_features.shape[1],), name='tabular_input')
        x_tab = layers.Dense(64, activation='relu')(tabular_input)
        x_tab = layers.Dense(32, activation='relu')(x_tab)
        
        # Input for the image data
        image_input = keras.Input(shape=(self.IMG_SIZE[0], self.IMG_SIZE[1], 1), name='image_input')
        x_img = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
        x_img = layers.MaxPooling2D(pool_size=(2, 2))(x_img)
        x_img = layers.Conv2D(64, (3, 3), activation='relu')(x_img)
        x_img = layers.MaxPooling2D(pool_size=(2, 2))(x_img)
        x_img = layers.Flatten()(x_img)
        
        # Combine the two branches
        combined = layers.concatenate([x_tab, x_img])
        
        # Final classifier layers
        z = layers.Dense(64, activation='relu')(combined)
        z = layers.Dropout(0.5)(z)
        output = layers.Dense(1, activation='sigmoid')(z)
        
        self.model = Model(inputs=[tabular_input, image_input], outputs=output)
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()
        return self.model
    
    def train_and_evaluate(self, epochs=10, patience=3):
        """Train and evaluate the multi-modal model"""
        if self.model is None or self.X_train_tab.empty or self.X_train_img.size == 0:
            print("Model or data not ready. Skipping training.")
            return

        print("\n" + "="*50)
        print("MODEL TRAINING AND EVALUATION")
        print("="*50)
        
        # Use early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            [self.X_train_tab, self.X_train_img],
            self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=self.BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        
    def generate_prediction_image(self):
        """
        Generates an image showing the prediction on an actual test sample.
        The image is saved to a file named 'prediction_output.png'.
        """
        print("Generating final prediction image...")
        
        # Make a single prediction on a random test sample
        if self.X_test_img.size > 0 and not self.X_test_tab.empty:
            test_index = np.random.randint(0, len(self.X_test_img))
            sample_img = np.expand_dims(self.X_test_img[test_index], axis=0)
            sample_tab = np.expand_dims(self.X_test_tab.iloc[test_index], axis=0)
            
            prediction = self.model.predict([sample_tab, sample_img])[0][0]
            
            # Determine prediction and confidence
            if prediction > 0.5:
                predicted_label = "Positive"
                confidence = prediction * 100
            else:
                predicted_label = "Negative"
                confidence = (1 - prediction) * 100
        else:
            predicted_label = "Unknown"
            confidence = 0.0
            
        # Now use matplotlib to display and save this image with text
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#1e1e1e')
        
        # Display the actual test image data instead of a placeholder rectangle
        # Remove the last dimension if it's a single-channel grayscale image
        img_data_to_show = self.X_test_img[test_index].squeeze()

        # Check if the image data has values
        if img_data_to_show.max() > 0:
            # Perform min-max scaling to stretch the contrast
            img_min = img_data_to_show.min()
            img_max = img_data_to_show.max()
            scaled_img = (img_data_to_show - img_min) / (img_max - img_min)
            ax.imshow(scaled_img, cmap='gray', vmin=0, vmax=1)
        else:
            # If the image is all black, just show it as is
            ax.imshow(img_data_to_show, cmap='gray', vmin=0, vmax=1)
        
        ax.set_title("Model Prediction", color='white', fontsize=16)
        
        # Add the prediction text
        ax.text(0.05, 0.90, f"Predicted: {predicted_label}", color='lime', fontsize=14, transform=ax.transAxes)
        ax.text(0.05, 0.85, f"Confidence: {confidence:.2f}%", color='lime', fontsize=12, transform=ax.transAxes)
        
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('prediction_output.png', dpi=300)
        plt.close()
        print("Image saved as 'prediction_output.png'.")
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*60)
        print("MULTI-MODAL PANCREATIC CANCER EARLY DETECTION")
        print("="*60)
        
        self.load_and_preprocess_data()
        
        # Add a check to ensure data was loaded before proceeding
        if self.tabular_features is None or self.tabular_features.empty or self.image_data.size == 0:
            print("Analysis aborted due to missing data.")
            return
            
        self.split_data()
        self.build_multimodal_model()
        self.train_and_evaluate()
        self.generate_prediction_image()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)

# Main execution
if __name__ == "__main__":
    # Initialize the analysis pipeline
    analysis = PancreaticCancerML()
    
    # Run complete analysis
    try:
        analysis.run_complete_analysis()
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("Please ensure all required packages are installed:")
        print("pip install pandas numpy matplotlib scikit-learn tensorflow opencv-python tqdm nibabel")
