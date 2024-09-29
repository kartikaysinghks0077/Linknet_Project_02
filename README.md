#Satellite Image Segmentation with LinkNet
This project implements a deep learning model for segmenting satellite images using a LinkNet architecture. The dataset contains satellite images and corresponding masks with four different classes of objects: forest, vegetation, urban areas, and water bodies. The segmentation is performed using a custom multi-class LinkNet model trained on these images.

##Table of Contents
Introduction
Dataset
Model Architecture
Preprocessing
Training
Evaluation
Results
Installation
Usage


#Introduction
The goal of this project is to segment satellite images into four distinct classes:

Forest (#0f510d)
Vegetation (#00ff00)
Urban areas (#ffafaf)
Water bodies (#0000ff)

A LinkNet architecture is used for image segmentation, combined with custom losses like Dice Loss and Focal Loss to improve accuracy. The performance of the model is evaluated using metrics such as Intersection over Union (IoU) and accuracy.

Dataset
The dataset contains two folders:

images/: Satellite images in .png format.
masks/: Corresponding segmentation masks in .png format, where different pixel colors represent different object classes.
The masks are converted from RGB to a 2D label where:

Forest → 0
Vegetation → 1
Urban → 2
Water → 3
Model Architecture
The architecture is based on the LinkNet model, modified for multi-class segmentation. Key features include:

Convolutional layers for feature extraction.
MaxPooling and UpSampling for downscaling and upscaling.
Batch Normalization for stable training.
Dropout to prevent overfitting.
Loss Functions
Dice Loss: Measures the overlap between predicted and true masks.
Categorical Focal Loss: Focuses on hard-to-classify pixels.
Metrics
IoU (Jaccard Coefficient): Measures the intersection and union of predicted and true segmentation masks.
Accuracy: Measures the proportion of correct predictions.
Preprocessing
The preprocessing steps involve:

Loading Images and Masks: Images and masks are loaded and converted to numpy arrays.
Class Conversion: RGB masks are converted to a 2D label array where each unique color is assigned to a class.
Train-Test Split: The data is split into training and testing sets using an 80-20 split.
Training
The model is trained for 200 epochs with a batch size of 16.
The optimizer used is Adam.
Loss functions used are a combination of Dice Loss and Focal Loss.
The model is compiled with the following parameters:

python
model.compile(optimizer='adam', loss=total_loss, metrics=['accuracy', jacard_coef])
Model Training Command:
python

history = model.fit(X_train, y_train, 
                    batch_size=16, 
                    epochs=200, 
                    validation_data=(X_test, y_test),
                    shuffle=False)
#Evaluation
The model is evaluated on the test set using:

IoU (Jaccard Coefficient): Calculated using Keras' MeanIoU function.
Accuracy: Average accuracy across all classes.
Evaluation Results:

python

Mean IoU = <calculated IoU>
Accuracy = <calculated accuracy>
Results
The results include:

Training Loss vs Validation Loss plot
Training IoU vs Validation IoU plot
Sample predictions are visualized:

Test Image
Ground Truth Mask
Predicted Mask
Example visualizations:

python

plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)

plt.subplot(232)
plt.title('Ground Truth')
plt.imshow(ground_truth_color)

plt.subplot(233)
plt.title('Prediction')
plt.imshow(predicted_img_color)
plt.show()






python

from google.colab import drive
drive.mount('/content/drive')
Install segmentation models:


Prepare Dataset:

Store your satellite images in images/ and corresponding masks in masks/.
Train the Model:

Run the script to start training:
python
Copy code
model.fit(X_train, y_train, batch_size=16, epochs=200, validation_data=(X_test, y_test))
Evaluate the Model:

Load the saved model:
python
Copy code
from keras.models import load_model
model = load_model('models/satellite_standard_linknet_with_500epochs496*496.hdf5')
Visualize Predictions:

Use the provided visualization code to compare ground truth and predicted masks.
Conclusion
This project successfully segments satellite images into multiple classes using the LinkNet architecture. The combination of Dice Loss and Focal Loss enhances segmentation performance. Further improvements can include trying different architectures, adding more classes, or augmenting the dataset.

