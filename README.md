# Predicting Data using Machine Learning

I'm gonna talk to you about what goes on behind the code to give you an understanding of how it works. We can divide our code into 3 parts. We first have the Pre-Processing of the data, then we have the Machine Learning aspect of it where we train the model using the data and finally, we have the prediction where we give x data into the model to return a prediction after training the model with a pre-existing data set. 

## Pre-Processing

We can think of DataFrames as a 2D structure or like a 2D array of kinds that has a row and a column. We use a DataFrame here to load the .csv data to be used by the program. We can use this with the help of the pandas library.
```
import pandas as pd
df = pd.read_csv('sampleDataset.csv')
```
There are a few things that we do with this DataFrame before we're ready to train our model. This is called pre-processing. During pre-processing, we help clean our dataset. This helps is providing a more accurate and clean model execution.  

We use the shape function from pandas to return the number of rows and colums in the DataFrame.
`df.shape`

The dropna function can be used to trim the empty data. Dropna is used to drop rows where at least one row is missing.
`df.dropna()`
After this, we can compare the value from the original DataFrame that we got using shape function with the value from the shape function after using Dropna. We can find that there's almost some reduction in the number of rows in the DataFrame now. 

Now we check the dataset. If there are Yes and No values in the columns, we replace them with 1's and 0's since models understand only numbers. To do so, we use the replace function.
`df['col_name'].replace({'No': 0, 'Yes': 1}, inplace  =  True)`

## Training

Now that we're done with pre-processing the data, we can now train our model. Before that we need to split our data into training data and testing data. To do so we use the train_test_split() function from sklearn model selection. 

```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  =  train_test_split(x, y, test_size=0.20, random_state=42)
```

In the above line, we split the data into training and testing data where 80% of the data is used for training while 20% is used for testing.
We also shuffle our data to prevent any bias.

What is **bias**?
We can think of Bias as the accuracy in our predictions. If there exists a high bias, it can cause our prediction to be inaccurate. 

_"Bias is the  algorithm's tendency to consistently learn the wrong thing by not taking into account all the information in the data(underfitting)."_ - Foreman

Parametric algorithms are prone to high bias. A parametric algorithm has a set number of parameters it considers to train the data. Example of high-bias algorithms are Linear Regression, Linear Discriminant Analysis, and Logistic Regression.
 
 After this we can build our model with any of the ML algorithms we choose.
 
 `fit()` is implemented by every estimator and it accepts an input in its parameters like the sample data and its argument for labels. It can also take additional parameters like weights etc. 
 The fit method typically start with clearing any attributes already stored on the estimator and then perform parameter and data validation. They also are responsible for estimating the attr. out of the input data and store the model attr. and finally return the fitted estimator. 

Now to evaluate how accurate our model is, we can use different metrics but here we'll use accuracy score.

[img]https://bit.ly/2T5Grf1[/img]

Accuracy Score produces a result according to the sum of the number of times our model predicted no correctly(True Negative) and yes correctly(True Positive) by the total number of predictions.
```
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
```
The higher the accuracy score is, the better and more accurate our model is and so are our predictions. 