# BlurDetection
Machine Learning code to predict if an image is blurred or undistorted

The dataset used in the code is CERTH_ImageBlurDataset

Use the CSV files attached along with the directory to get the accuracy measure. The name of the files are "DigitalBlur_modified.csv" and "NaturalBlurSet_modified.csv".

# ABOUT THE MODEL USED:

- The model is made using Support Vector Machines. 

- The training-test set accuracy is 86.33%. The accuracy for the evaluation set is 83.17%.

- The feature engineering done, mainly uses three kernel filters, namely Laplace, Sobel and Roberts.
