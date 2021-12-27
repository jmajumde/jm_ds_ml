# >>>> Question1:
"""
Decision Tree - Bank Marketing Dataset
Description
You are given the 'Portuguese Bank' marketing dataset which contains data about a telemarketing campaign run by the bank to sell a product (term deposit - a type of investment product).

Each row represents a 'prospect' to whom phone calls were made to sell the product. There are various attributes describing the prospects, such as age, profession, education level, previous loans taken by the person etc. Finally, the target variable is 'purchased' (1/0), 1 indicating that the person had purchased the product. A sample of the training data is shown below (note that 'id' shouldn't be used to train the model) :

    age          job  marital          education default  housing     loan  \
0   30  blue-collar  married           basic.9y      no      yes       no
1   39     services   single        high.school      no       no       no
2   25     services  married        high.school      no      yes       no
3   38     services  married           basic.9y      no  unknown  unknown
4   47       admin.  married  university.degree      no      yes       no

     contact month day_of_week ...  pdays  previous     poutcome  \
0   cellular   may         fri ...    999         0  nonexistent
1  telephone   may         fri ...    999         0  nonexistent
2  telephone   jun         wed ...    999         0  nonexistent
3  telephone   jun         fri ...    999         0  nonexistent
4   cellular   nov         mon ...    999         0  nonexistent

   purchased  id
0          0   1
1          0   2
2          0   3
3          0   4
4          0   5

As an analyst, you want to predict whether a person will purchase the product or not. This will help the bank reduce their marketing costs since one can then target only the prospects who are likely to buy.

Build a decision tree with default hyperparameters to predict whether a person will buy the product or not.

The training data is provided here:
/data/training/bank_train.csv

After you train the model, use the test data to make predictions. The test data can be accessed here.
/data/test/bank_test.csv

You have to write the predictions in the file
/code/output/bank_predictions.csv

in the following format (note the column names carefully):
     bank_predicted    id
0               0  2041
1               0   399
2               0  1400
3               0  3709
4               0  2111


Datasets
Training dataset -> https://media-doselect.s3.amazonaws.com/generic/BNnYdvopPM53yVw2yNOaWkj7Z/training-training-bank_train.zip
Validation dataset -> https://media-doselect.s3.amazonaws.com/generic/NByMNGB5jVPRe8k41qeWWnvA/validation-validation-bank_test.zip

""""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import metrics, preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# read training dataset
bank_train = pd.read_csv("/home/jmajumde/PGDDS-IIITB/ML-2/DecisionTree/CodingPractice_data/bank_train.csv")

# read test set
bank_test = pd.read_csv("/home/jmajumde/PGDDS-IIITB/ML-2/DecisionTree/CodingPractice_data/bank_test.csv")

print(bank_train.head(5))
print("=====================")
print(bank_test.head(5))

# Build the model
# Create a decision tree object with default hyperparameters
tree =  DecisionTreeClassifier()

# Train the model
print(bank_train.columns)

# Create x_train: Drop the columns 'purchased' (target) and 'id'
x_train = bank_train.drop(['purchased','id'], axis=1)

# Create y_train
y_train = bank_train['purchased']

# Fit the model
tree.fit(x_train,y_train)

# Make predictions using test data
predictions = tree.predict(bank_test.drop(['id'], axis=1))
print(predictions[:5])

# Write the columns 'id' and 'predictions' into the output file
d = pd.DataFrame({'id': bank_test['id'], 'bank_predicted': predictions})
d.head()
# Write the output
d.to_csv('/home/jmajumde/PGDDS-IIITB/ML-2/DecisionTree/CodingPractice_data/output/bank_predictions.csv', sep=",")

# >>>>> Question2:
"""
In the previous question on this dataset, you had built a decision tree with default hyperparameters. In this question, you will find the optimal value of the hyperparameter max_depth usingGridSearchCV(), and then build a model using the optimal value of max_depth to predict whether a given prospect will buy the product.

To find the optimal value, you can plot training and test accuracy versus max_depth using matplotlib (the code is already written - you will see the plot displayed below the coding console).
"""
# split into x_train and y_train
x_train = bank_train.drop(['purchased', 'id'], axis=1)
y_train = bank_train[['purchased']]

# Hyperparameter tuning: maxdepth
# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on: specify a range of max_depth
parameters = {'max_depth': range(1,10)}

dtree = DecisionTreeClassifier()
# fit tree on training data
tree = GridSearchCV(dtree, parameters,
                     cv=n_folds,
                   scoring="accuracy",
                   return_train_score=True)
tree.fit(x_train, y_train)
scores = tree.cv_results_
print(pd.DataFrame(scores).head(5))

# plotting accuracies with max_depth (code already written)
plt.figure()
plt.plot(scores["param_max_depth"],
         scores["mean_train_score"],
         label="training accuracy")
plt.plot(scores["param_max_depth"],
         scores["mean_test_score"],
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
plt.savefig('/home/jmajumde/PGDDS-IIITB/ML-2/DecisionTree/CodingPractice_data/output/hyperparam.svg')










######
