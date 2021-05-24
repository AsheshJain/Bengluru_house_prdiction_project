# Bengluru House Price Prediction Project
This project is a Machine learning Project which is trained on a kaggle's dataset which is also present in this repositery.
As the name suggests,this Project predicts the price of the price of the House depends on the following factors:
* Total Area(in Square Feet) of the house.
* Number of bathrooms in it.
* Number of Balconys in it.
* Number of Bedrooms or How much BHK is the House.
* Type of Area in which House is.
* Location of the House

## Work Flow of the Project
* ### Data Cleaning And Data Analysing
  This is the main aspect of any Data science Project as the data is very messy such as : the 'size' column contains different values with some where BHK written or somewhere Bedroom is written
  so we need make it uniform. similarly in 'total_sqft' column too, it have values that are ranges like 2240-4490 so we need to make number and i did it by taking the average of the ranges.
  There are a lot more tasks like this these that need to handled by us to proceed further.
* ### Data Preprocessing or Data Preparing For Model
  This is all about Making the data ready for the machine learning Model. In this Project one of the important task to be done is to remove the outliners i.e the values which lies far from the majority of the values, these values affect the model predictions in a very negetive ways as they unnessesary expand the ranges.
  Sometimes our data contains the values that are unexpected pratically like the number of bathrooms are mcuh greater than the number of bedrooms , or total area is too small for the size of house. We need to analyse data in both practical and scientific aspects before fitting it into the model.
  In our data set in 'location' column we have many values that are arrrived only once so we assign all those to the 'others' location name because they are the outliners which cause problem.
  At the last we need to Encode those columns which have string values, as Model only works wiht the numerical values.In our dataset those columns that were encoded are 'location' and 'areas'.
* ### Model Selection And Testing 
  There are many types of machine learning algorithms models to be applied on the given problem we need to find the most optimum one so we try various models and test them and comapare their scores to choose the best one. We also need to check the different parameter values for the same model and choose the best parameter this process is often called model tuning.
* ### Presenting The Model And Predicting The Values
  The final step is to present the model or to deploy the Model so that we can check its implementation and test it on some practical situations.
  
## Libraries Used
* PANDAS
* NUMPY
* DASH
* SKLEARN
* OS
* MATPLOTLIB
* PLOTLY

## Modules Imported And Their Uses
* #### import pandas as pd 
to do all the data manupulation and data analysis.
* #### import numpy as np
to do the numerical calculations on data
* #### import matplotlib.pyplot as plt
to visualise the data to do better analysis
* #### from sklearn.model_selection import train_test_split
to split the data into testing and training datasets
* #### from sklearn.linear_model import LinearRegression
Linear Regression Machine learning Model present in sklearn library
* #### from sklearn.linear_model import Lasso
this is L1 normalization model which we often called Lasso model
* #### from sklearn.tree import DecisionTreeRegressor
this is Decision Tree Model present in sklearn library
* #### from sklearn.linear_model import Ridge
this is L2 normalization model which we often called Ridge model
* #### from sklearn.model_selection import ShuffleSplit
to shuffle the data so that we can split out data in training and testing in better way
* #### from sklearn.model_selection import cross_val_score
this is k-fold cross validation to check the accuracy of model more precisely
* #### from sklearn.model_selection import GridSearchCV
this is used to compare the different Models and along with their different parameters and to get the results in a much nice format for better comparison.
* #### import dash
* #### import dash_core_components as dcc
* #### import dash_html_components as html
* #### import plotly.graph_objs as go
* #### from dash.dependencies import Input, Output
#### these all are used to make the Dashboard through which i have presented the model

## Model Working Shots :
### At a Glance :
![Screenshot 2021-05-24 at 12 03 51 PM](https://user-images.githubusercontent.com/78098555/119306880-776ae380-bc88-11eb-8f48-77c308eef7fd.png)


### After Prediction :
![Screenshot 2021-05-24 at 12 04 19 PM](https://user-images.githubusercontent.com/78098555/119306763-468aae80-bc88-11eb-9e8a-e0e5bd386c46.png)

