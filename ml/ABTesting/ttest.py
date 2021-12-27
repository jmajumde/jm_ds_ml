# StatisticsEDA/statistical_methods_for_machine_learning/statistical_methods_for_machine_learning.pdf
import numpy as np
from scipy.stats import ttest_ind

np.random.seed(10)

# generate two independent samples
data1=10 * np.random.randn(100) + 50 # adding noise
data2=101 * np.random.randn(100) + 52 # adding noise
mystat, p = ttest_ind(data1, data2)
print("mystat: %.3f, p: %.3f" % (mystat,p))

alpha = 0.05
if p > alpha:
    print('Same dist, fail to reject H0')
else:
    print('diff dist, reject H0')
