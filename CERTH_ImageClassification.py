#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries

import numpy as np
import pandas as pd
import os
# print(os.listdir("../input"))
import matplotlib.pyplot as plt
import cv2
import sklearn
import seaborn as sb

from skimage.color import rgb2gray
from skimage.filters import laplace, sobel, roberts


# In[13]:


#TRAINING MODEL

#input paths of training set

print('Taking inputs for the Training Set. So please add the inputs in the code itself.')

#Artificially-Blurred
# ab_path ='../input/artificially-blurred/Artificially-Blurred/'
ab_path = input("Enter path for Artificially-Blurred images: ") 

#Naturally-Blurred
# nb_path='../input/naturally-blurred/Naturally-Blurred/'
nb_path = input("Enter path for Naturally-Blurred images: ") 

#Undistorted
# ud_path ='../input/undistorted/Undistorted/'
ud_path = input("Enter path for Undistorted images: ") 



ab_images = os.listdir(ab_path)
nb_images = os.listdir(nb_path)
undistorted = os.listdir(ud_path)


# In[14]:


#Evaluation Model

#input paths of the Evaluation set

print('Taking inputs for the Evaluation Set. So please add the inputs in the code itself.')

#Digital_Blur
# db_ES_path= '../input/digitalblurset-evaluationset/DigitalBlurSet/' 
db_ES_path = input("Enter the path for DigitalBlurSet: ")

#Natural_Blur
# nb_ES_path= '../input/naturalblurset-evaluationset/NaturalBlurSet/' 
nb_ES_path = input("Enter the path for NaturalBlurSet: ")

#importing the sheets having the correct label for accuracy measure

print('Do not forget to input the paths of the Modified CSV which are added in the ZIP folder')

pthToCSVDBY = input("Enter the path for DigitalBlurModified CSV file: ")
dbY=  pd.read_csv(pthToCSVDBY)

# dbY=  pd.read_csv('../input/digitalblures/DigitalBlur_modified.csv')
# dbY.drop('Image Name', axis= 1, inplace= True)
# print(dbY.shape)


pthToCSVNBY = input("Enter the path for NaturalBlurModified CSV file: ")
nbY=  pd.read_csv(pthToCSVNBY)

# nbY=  pd.read_csv('../input/esmodified-123/NaturalBlurSet_modified.csv')
# nbY.drop('Image Name', axis= 1, inplace= True)
# print(nbY.shape)

db_ES = os.listdir(db_ES_path)
nb_ES= os.listdir(nb_ES_path)


# In[11]:


#function to get the features

def get_data(path,images):
    features=[]
    for img in images:
        feature=[]
        image_gray = cv2.imread(path+img,0)
        lap_feat = laplace(image_gray)
        sob_feat = sobel(image_gray)
        rob_feat = roberts(image_gray)
        feature.extend([img,lap_feat.mean(),lap_feat.var(),np.amax(lap_feat),
                        sob_feat.mean(),sob_feat.var(),np.max(sob_feat),
                        rob_feat.mean(),rob_feat.var(),np.max(rob_feat)])
        
        features.append(feature)
    return features


# In[ ]:


#Training Set

print('I am exracting the features from the Training set now. It might take a while, approximately 20-30 minutes, please relax till then')

#feature engineering

ab_images_features = get_data(ab_path,ab_images)
nb_images_features = get_data(nb_path,nb_images)
undistorted_features = get_data(ud_path,undistorted)


# In[ ]:


#Training Data
#converting the features in a dataframe and visualising them

print('Congrats! I have extracted the feature. It is now my time to learn some important things. Now I am making the dataframe of extracted information')

ab_df = pd.DataFrame(ab_images_features)
ab_df.drop(0,axis=1,inplace=True)
# ab_df.head()

nb_df = pd.DataFrame(nb_images_features)
nb_df.drop(0,axis=1,inplace=True)
# nb_df.head()

undistorted_df = pd.DataFrame(undistorted_features)
undistorted_df.drop(0,axis=1,inplace=True)
# undistorted_df.head()


# In[ ]:


#Training Data
#to plot and visualise the data
# this is for plotting purpose

print('plotting the data for your reference')


label = ['Artificially_Blurred','Naturally_Blurred','Undistorted']
no_images=[len(ab_images_features),len(nb_images_features),len(undistorted_features)]

def plot_bar_x():
    index = np.arange(len(label))
    plt.bar(index, no_images)
    plt.xlabel('Image_type', fontsize=10)
    plt.ylabel('No of Images', fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=0)
    plt.title('Data Visualization')
    plt.show()
plot_bar_x()


# In[ ]:


#Evaluation Set

print('I think I should extract the features of the Evaluation Set as well. This will take some time again. Sit back and relax.')

#feature engineering

db_ES_features= get_data(db_ES_path, db_ES)
nb_ES_features= get_data(nb_ES_path, nb_ES)


#converting the features into a dataframe

print('Extracted the data from evaluation set as well. Making the dataframes from them.')

db_ES_df= pd.DataFrame(db_ES_features)
# print(db_ES_df.head())

nb_ES_df= pd.DataFrame(nb_ES_features)
# print(nb_ES1_df.head())

ES= pd.DataFrame()
ES= ES.append(db_ES_df)
ES= ES.append(nb_ES_df)

ES.sort_values(by=0, inplace=True)
filename= ES[0].values

ES.drop(0, axis=1, inplace=True)

x_ES_feat= np.array(ES)



#importing the sheets having the correct label for accuracy measure

# dbY=  pd.read_csv('C:/Users/Lenovo/Downloads/Compressed/CERTH_ImageBlurDataset/EvaluationSet/DigitalBlur_modified.csv')
# # dbY.drop('Image Name', axis= 1, inplace= True)
# # print(dbY.shape)

# nbY=  pd.read_csv('C:/Users/Lenovo/Downloads/Compressed/CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet_modified.csv')
# # nbY.drop('Image Name', axis= 1, inplace= True)
# # print(nbY.shape)

Y_ES= pd.DataFrame()
Y_ES= Y_ES.append(dbY)
Y_ES= Y_ES.append(nbY)

Y_ES.sort_values(by=['Image Name'], inplace=True)

Y_ES.drop('Image Name', axis= 1, inplace= True)


original_label= Y_ES['Blur Label'].values

y_ES= Y_ES.values


# In[ ]:


#FITTING THE MODEL
#SUPPORT VECTOR MACHINES

print('Now I am learning from the training data, please let me learn so that I can do the best.')

#training

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, classification_report
# from keras.utils import to_categorical
images=pd.DataFrame()

images = images.append(undistorted_df)
images = images.append(nb_df)
images = images.append(ab_df)
all_features = np.array(images)
y_f = np.concatenate((np.zeros((undistorted_df.shape[0], ))-1, np.ones((nb_df.shape[0], )), 2*np.ones((ab_df.shape[0], ))-1), axis=0)

x_train,x_valid,y_train,y_valid = train_test_split(all_features,y_f,test_size=0.3,stratify=y_f)



svm_model = svm.SVC(C=100,kernel='rbf')
svm_model.fit(x_train,y_train)
pred =svm_model.predict(x_valid)
print('Accuracy:',accuracy_score(y_valid,pred))
print('Confusion matrix:\n',confusion_matrix(y_valid,pred))


# y_valid_cat = to_categorical(y_valid, num_classes=3)
# pred_cat = to_categorical(pred, num_classes=3)
# print('F1_score:',f1_score(y_valid_cat,pred_cat, average='weighted'))
# print('Classification_report:\n',classification_report(y_valid_cat,pred_cat))

print('Results from the test set that was split in the training set itself')
print(pred)


# In[ ]:


#getting the predictions from the model

#Evaluation

print('Time for the real test. Now I am running the evaluation set.')

pred_ES =svm_model.predict(x_ES_feat)
print('Accuracy:',accuracy_score(y_ES,pred_ES))
print('Confusion matrix:\n',confusion_matrix(y_ES,pred_ES))

print('I think I have done well! But what do you think, is it good enough? Honestly, I have worked real hard for this.')

# y_valid_cat_ES = to_categorical(y_ES, num_classes=3)
# pred_cat_ES = to_categorical(pred_ES, num_classes=3)
# print('F1_score:',f1_score(y_valid_cat_ES,pred_cat_ES, average='weighted'))
# print('Classification_report:\n',classification_report(y_valid_cat_ES,pred_cat_ES))


#reshaping the variables for better visualisation

print('One last gift, I have saved my predictions in a CSV file named RESULTS. It contains both the predicted values as well as original labels for your reference.')

filename.reshape(1480, 1)
pred_ES.reshape(1480,1), 
original_label.reshape(1480, 1)

d1= pd.DataFrame({'name':filename,'original label':original_label, 'predictions':pred_ES})
# d1.head()

d1.to_csv('RESULTS')

print('I did all the things I was made to do. See you! Bye!')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




