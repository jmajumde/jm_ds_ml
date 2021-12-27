import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
#cancer
print(cancer.DESCR)
print(len(cancer.target))

malignant=len(cancer.data[cancer.target==0])
benign=len(cancer.data[cancer.target==1])
print("benign:" + str(benign))
print("malignant:" + str(malignant))


cancer_df=pd.DataFrame(cancer.data,columns=cancer.feature_names)
cancer_df.head()
cancer_df.columns

feature_mean=list(cancer_df.columns[0:10])
feature_worst=list(cancer_df.columns[20:31])
