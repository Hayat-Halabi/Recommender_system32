# Recommender_system32
Recommender systems ws32
``` python
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import math  # For mathematical functions
import itertools  # For creating iterators for efficient looping

# Import modeling helpers
from sklearn.preprocessing import Normalizer, scale  # For data preprocessing
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.metrics import confusion_matrix  # For evaluating classification results
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score  # For hyperparameter tuning and cross-validation

# Import evaluation metrics for regression
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error

# Import evaluation metrics for classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import visualization libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns





















```
Links to access data
```
movies_csv:
https://vocproxy-1-9.us-west-2.vocareum.com/files/home/labsuser/movies.csv?_xsrf=2%7C23e987d6%7C3998626cb435d6edd51bde567dfcb338%7C1709519232
ratings_csv:
 https://vocproxy-1-9.us-west-2.vocareum.com/files/home/labsuser/ratings.csv?_xsrf=2%7C23e987d6%7C3998626cb435d6edd51bde567dfcb338%7C1709519232
```
