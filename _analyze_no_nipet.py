import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Function to load DICOM PET images
def load_dicom_pet_images(dicom_dir):
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    dicom_files.sort()  # Sort files to ensure correct order
    pet_images = []
    for file in dicom_files:
        try:
            dicom_data = pydicom.dcmread(file)
            if hasattr(dicom_data, 'pixel_array'):
                pet_image = dicom_data.pixel_array.astype(float)
                pet_images.append(pet_image)
            else:
                st.warning(f'File {file} does not contain pixel data.')
        except Exception as e:
            st.error(f'Error processing file {file}: {str(e)}')
    return np.stack(pet_images, axis=-1) if pet_images else np.array([])

# Function to calculate SUV metrics (placeholder)
def calculate_suv_metrics(pet_image_data):
    suv_max = np.max(pet_image_data)
    suv_mean = np.mean(pet_image_data)
    suv_values = np.random.rand(10)  # Placeholder for SUV values
    return suv_max, suv_mean, suv_values

# Function to preprocess images for classification
def preprocess_images(images):
    images_resized = []
    for img in images:
        if img.ndim == 2:  # If the image is 2D, add a channel dimension
            img = np.expand_dims(img, axis=-1)
        if img.ndim == 3:  # Now it should be 3D
            img_resized = tf.image.resize(img, [128, 128])
            images_resized.append(img_resized)
    images_resized = np.array(images_resized).astype(np.float32)
    return images_resized

# Function to build and train the CNN model
def build_and_train_model(X_train, y_train, X_test, y_test):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')  # Assuming binary classification for PET images
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    return model, test_acc

# Main Streamlit app code
def main():
    st.header('PET Image Analysis and Classification')

    # Sidebar for file upload and parameter inputs
    st.sidebar.header('Upload DICOM Files')
    uploaded_files = st.sidebar.file_uploader('Upload DICOM files', accept_multiple_files=True)

    if uploaded_files:
        st.sidebar.subheader('Selected DICOM Files:')
        for file in uploaded_files:
            st.sidebar.text(file.name)

        # Temporary directory to store uploaded files
        temp_dir = './temp_dicom'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Save uploaded files to temporary directory
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, 'wb') as f:
                f.write(file.read())

        # Load DICOM PET images from temporary directory
        pet_image_data = load_dicom_pet_images(temp_dir)

        # Check if pet_image_data is not empty
        if len(pet_image_data) == 0:
            st.error('No DICOM PET images found or could not be loaded.')
            return

        # Example PET image display (Axial slice)
        st.write('Axial Slice of DICOM PET Image')
        iz = st.slider('Select Slice Index', 0, pet_image_data.shape[2] - 1, pet_image_data.shape[2] // 2)

        if st.checkbox('Quantification'):
            # Perform SUV calculation and display metrics
            suv_max, suv_mean, suv_values = calculate_suv_metrics(pet_image_data[:, :, iz])
            st.subheader('Quantification Results')
            st.write(f'SUV max: {suv_max:.2f}')
            st.write(f'SUV mean: {suv_mean:.2f}')
            # Display bar chart or table of SUV values
            st.bar_chart(suv_values)

        # Create a Matplotlib figure and plot the PET image slice
        fig, ax = plt.subplots()
        ax.imshow(pet_image_data[:, :, iz], cmap='hot', interpolation='nearest')
        ax.set_title(f'Axial Slice {iz}')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.grid(True)
        
        # Display the plot using st.pyplot() with the figure object
        st.pyplot(fig)

        # Further processing and classification
        if st.checkbox('Run PET Image Classification'):
            # Preprocess images
            images = [pet_image_data[:, :, i] for i in range(pet_image_data.shape[2])]
            images = preprocess_images(images)
            labels = np.random.randint(0, 2, size=len(images))  # Placeholder for labels

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

            # Build and train the CNN model
            model, test_acc = build_and_train_model(X_train, y_train, X_test, y_test)
            st.write(f'Test accuracy: {test_acc}')

            # Plot some predictions
            predictions = model.predict(X_test)

            def plot_image(i, predictions_array, true_label, img):
                predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
                plt.grid(False)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(np.squeeze(img), cmap=plt.cm.binary)
                predicted_label = np.argmax(predictions_array)
                color = 'blue' if predicted_label == true_label else 'red'
                plt.xlabel(f"{predicted_label} ({100 * np.max(predictions_array):2.0f}%) ({true_label})", color=color)

            num_rows = 5
            num_cols = 3
            num_images = min(num_rows * num_cols, len(X_test))  # Ensure we don't exceed the number of test images
            plt.figure(figsize=(2 * num_cols, 2 * num_rows))
            for i in range(num_images):
                plt.subplot(num_rows, num_cols, i + 1)
                plot_image(i, predictions, y_test, X_test)
            plt.tight_layout()
            st.pyplot(plt)

# Run the main function
if __name__ == '__main__':
    main()
