"""
The String Game
Description
Sanskar and Mahima are playing the String Game.

Rules of the game are as follows.

Both players are given the same string, S.
Both players have to make substrings using the letters of the string S.
Sanskar has to make words starting with consonants.
Mahima has to make words starting with vowels.
The game ends when both players have made all possible substrings.

Scoring of the game is done by the following rule

A player gets +1 point for each occurrence of the substring in the string S.

For Example:

S = BANANA

Mahima's vowel beginning word = ANA
Here, ANA occurs twice in BANANA. Hence, Mahima will get 2 Points.  ﻿

Sanskar wins

Your task is to determine the winner and their score.

Sample Input      : BANANA
Sample Output : Sanskar 12

Please note that this question was asked in a Data Analyst interview.
"""
import numpy as np
import pandas as pd

# Read the input
#S = input()
S = 'BANANA'
# Write your code below
mylist=list(S)
Sanskar=0
Mahima=0
vowelList=['a','e','i','o','u','A','E','I','O','U']
for i in range(0,len(mylist)):
    if mylist[i] in vowelList:
        Mahima=Mahima+len(mylist)-(i+1)+1
        #print(Mahima)
    else:
        Sanskar=Sanskar+len(mylist)-(i+1)+1
        #print(Sanskar)

if Mahima > Sanskar:
    print('Mahima ' + str(Mahima))
elif Mahima == Sanskar:
    print('Draw')
    print('Mahima ' + str(Mahima))
    print('Sanskar ' +  str(Sanskar))
else:
    print('Sanskar ' +  str(Sanskar))


"""
Identity Cards
Description
You have been asked to ensure that the first and last names of employees in an organization begin with a capital letter in their identity cards. For example, akshay ginodia should be capitalised correctly as Akshay Ginodia.

akshay ginodia  => Akshay Ginodia

Given a full name, your task is to capitalize the name appropriately.

Input Format

A single line of input containing the full name, S.

Constraints

0 < len(S) < 100

The string consists of alphanumeric characters and spaces.
Note: in a word only the first character is capitalized. Example 12abc, when capitalized, remains 12abc.

Output Format
Print the capitalized string, S.
Sample Input: akshay ginodia
Sample Output: Akshay Ginodia
"""
s = 'the brown fox'
lst = [word[0].upper() + word[1:] for word in s.split()]
print(lst)
s = " ".join(lst)
print(s)

S = "akshay 123ginodia"
if S != "" and len(S) < 100:
    #print(S)
    #lst = [w[0].upper() + w[1:] if w[0].isalpha() for w in S.split()]
    lst=[]
    for w in S.split():
        if w[0].isalpha():
            str = w[0].upper() + w[1:]
        else:
            str = w[0:]
        lst.append(str)
    output=" ".join(lst)
    print(output)


"""
Binary Representation
Description
Given a positive integer 'n', print the binary representation of the number.

Constraint
n < 10000

Input format: A single integer denoting 'n'
Output format: A string denoting the binary representation of the number 'n'

Sample Input  : 19
Sample Output : 10011

Use the fewest character possible characters to display the output. For instance, if '1' is input, the output should be '1'. Do not provide '0001' or '01' as the output.

You can learn about binary representation here https://www.geeksforgeeks.org/digital-logic-binary-representations/

Please note that this question was asked in a Data Analyst interview.
"""
n=3
if n < 10000:
    print("{0:b}".format(n))
    

"""
Alphabetic patterns
Description
Given a positive integer 'n' less than or equal to 26, you are required to print the below pattern

Sample Input: 5

Sample Output :
--------e--------
------e-d-e------
----e-d-c-d-e----
--e-d-c-b-c-d-e--
e-d-c-b-a-b-c-d-e
--e-d-c-b-c-d-e--
----e-d-c-d-e----
------e-d-e------
--------e--------

Sample Input  : 3

Sample Output :
----c----
--c-b-c--
c-b-a-b-c
--c-b-c--
----c----

Please note that this question was asked in a Data Scientist interview.
"""




"""
Regularised Regression - I
Description
You are provided with the community and crime dataset.

Your task is to print the number of rows and name of columns. Assign the number of rows to a new variable named 'n_row' and the number of columns to a variable named 'n_column'.

The dataset can be found at 'https://media-doselect.s3.amazonaws.com/generic/ka2eL88aayAyB3W7eRVL1xWq/communities_final.csv'.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("https://media-doselect.s3.amazonaws.com/generic/ka2eL88aayAyB3W7eRVL1xWq/communities_final.csv")
n_row=df.shape[0]
n_column=df.shape[1]
print(n_row)
print(n_column)

"""
Regularised Regression - II
Description
So far in this exercise, you have completed the data importing on the provided dataset. You need to drop the non-predictive columns :

state
county
community
communityname
fold

Note that the missing values are denoted by '?' in the provided dataset.
Once you have deleted the non-predictive columns, drop all the columns that have missing values. Once you have performed the data cleaning exercise, store the names of the columns in a list named as update_column_name.
"""

import pandas as pd
import numpy as np

# Reading training data
df = pd.read_csv("https://media-doselect.s3.amazonaws.com/generic/QBb0aGyg597GONqxVapbA3Kd3/communities_final.csv")
df.shape
# Drop the non-predictive columns
df.drop(columns=['state','county','community','communityname','fold'], inplace=True)
df.shape

# Drop columns with NaN values
df = df.replace('?',np.nan).dropna(axis=1)
df.shape

n_row = df.shape[0]
n_column = df.shape[1]
print(n_row)
print(n_column)


"""
Regularised Regression - IV
Description
Now your task is to perform lasso regression on the complete dataset.
Note: All operations you performed in the previous question have already been performed on the dataset here. You can take any other measures to ensure a better outcome if you want. (for example: normalising or standardising any values or adding any other columns). The dataset has been divided into train and test sets and both have been loaded in the coding console.

You have to write the predictions in the file:
/code/output/predictions.csv
"""
import numpy as np
import pandas as pd

# Read training data

train = pd.read_csv("https://media-doselect.s3.amazonaws.com/generic/ELv39dyjbNdXeGr0wGb80b1jV/train%20(1).csv")
y_train = pd.read_csv("https://media-doselect.s3.amazonaws.com/generic/RNPx4L9kOjZ7JMAeEnvnpjYxL/train_y.csv")

# Read test data
test = pd.read_csv("https://media-doselect.s3.amazonaws.com/generic/oezG9OJgB47jrYajpwQojjaJJ/test.csv")

# Import the required libraries
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# Lasso Regression Model
lasso = Lasso()

folds = 5
# cross validation
model_cv = GridSearchCV(estimator = lasso,
                        param_grid = {'alpha':[0.001]},
                        scoring= 'neg_mean_absolute_error',
                        cv = folds,
                        return_train_score=True,
                        verbose = 1)
model_cv.fit(train,y_train)

y_test_predicted = model_cv.predict(test)
# Write the output
predictions = pd.DataFrame(y_test_predicted)
predictions.rename(columns= {0:'predictions'}, inplace=True)
#predictions.to_csv("predictions.csv")
predictions.head()





#######
