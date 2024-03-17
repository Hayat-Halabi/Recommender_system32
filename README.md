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

# Configure visualizations
%matplotlib inline
mpl.style.use('ggplot')
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette="dark", style='whitegrid', color_codes=True)

# Center all plots using IPython display
from IPython.core.display import HTML

# Apply CSS style to center plots
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""");

# Configure visualization parameters for improved appearance
params = {
    'axes.labelsize': "large",  # Set label font size for axes
    'xtick.labelsize': 'x-large',  # Set label font size for x-axis ticks
    'legend.fontsize': 20,  # Set font size for legend
    'figure.dpi': 150,  # Set DPI (dots per inch) for figure
    'figure.figsize': [25, 7]  # Set figure size
}

# Update Matplotlib parameters with the configured values
plt.rcParams.update(params)

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
df_r = ratings.copy()
df_m = movies.copy()
ratings.head()
ratings.shape
ratings.describe()

# Drop the 'timestamp' column from the 'ratings' DataFrame
ratings.drop(['timestamp'], axis=1, inplace=True)
ratings.head()

movies.head()
print('Shape: ', movies.shape, '\n')
movies.info()

df_combined = pd.merge(ratings, movies, on='movieId')
df_combined.shape
df_combined.head()


# Create an empty dictionary to store the count of different genre values
genres = {}

# Define a function to find and count genres in the dataset
def find_genres():
    # Iterate through each movie's genre information in the dataset
    for genre in movies['genres']:
        # Split the genre string into individual words using the '|' separator
        words = genre.split('|')

        # Iterate through each individual word in the split genre
        for word in words:
            # Increment the count of the genre in the genres dictionary
            # If the genre is already in the dictionary, increment its count by 1
            # If the genre is not in the dictionary, add it with a count of 1
            genres[word] = genres.get(word, 0) + 1

# Call the function to populate the genres dictionary with genre counts
find_genres()

genres

# Replace the genre key '(no genres listed)' with the key 'None'
# and update its corresponding value
genres['None'] = genres.pop('(no genres listed)')


# Create a DataFrame 'df_n_ratings' by grouping 'df_combined' by movie title and calculating the mean rating
df_n_ratings = pd.DataFrame(df_combined.groupby('title')['rating'].mean())

# Calculate the total number of ratings for each movie title and add it as a new column in 'df_n_ratings'
df_n_ratings['total ratings'] = pd.DataFrame(df_combined.groupby('title')['rating'].count())

# Rename the 'rating' column to 'mean ratings' in the 'df_n_ratings' DataFrame
df_n_ratings.rename(columns={'rating': 'mean ratings'}, inplace=True)

# Sort the 'df_n_ratings' DataFrame in descending order based on the 'total ratings' column,
# and retrieve the top 10 movies with the highest total ratings
df_n_ratings.sort_values('total ratings', ascending=False).head(10)

# Set the figure size for the plot
plt.figure(figsize=(8, 4))

# Create a distribution plot (histogram) of the 'total ratings' column from the 'df_n_ratings' DataFrame
# with 20 bins to represent the data distribution
sns.distplot(df_n_ratings['total ratings'], bins=20)

# Set the label for the x-axis
plt.xlabel('Total Number of Ratings')

# Set the label for the y-axis
plt.ylabel('Probability')

# Display the plot
plt.show()

# Sort the 'df_n_ratings' DataFrame in descending order based on the 'mean ratings' column,
# and retrieve the top 10 movies with the highest mean ratings
df_n_ratings.sort_values('mean ratings', ascending=False).head(10)

# Printing the total number of users who gave a rating of 5.0
print('Total no of users that gave rating of 5.0 : ', len(df_n_ratings.loc[df_n_ratings['mean ratings'] == 5]), '\n')

# Printing the total number of individual users who gave a rating of 5.0
print('Total no of Individual users that gave rating of 5.0 : ', len(df_n_ratings.loc[(df_n_ratings['mean ratings'] == 5)
                                                                           & (df_n_ratings['total ratings'] == 1)]))

# Creating a new figure for the plot with a specified size
plt.figure(figsize=(8, 4))

# Creating a distribution plot (histogram and kernel density estimate) of the 'mean ratings' column
sns.distplot(df_n_ratings['mean ratings'], bins=30)

# Adding a label to the x-axis of the plot
plt.xlabel('Mean Ratings')

# Adding a label to the y-axis of the plot
plt.ylabel('Probability')

# Displaying the plot
plt.show()

util_mat = df_combined.pivot_table(index='userId', columns='title', values='rating')

# Displaying the first 20 rows of the utility matrix.
util_mat.head(20)

user_util_matrix = util_mat.copy()

# Filling the NaN values in the utility matrix rows with the corresponding user's mean ratings.
# This step is necessary for calculating Pearson correlation, and it assumes average ratings for unrated movies.
user_util_matrix = user_util_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
user_util_matrix.head(5)





```
Links to access data
```
movies_csv:
https://vocproxy-1-9.us-west-2.vocareum.com/files/home/labsuser/movies.csv?_xsrf=2%7C23e987d6%7C3998626cb435d6edd51bde567dfcb338%7C1709519232
ratings_csv:
 https://vocproxy-1-9.us-west-2.vocareum.com/files/home/labsuser/ratings.csv?_xsrf=2%7C23e987d6%7C3998626cb435d6edd51bde567dfcb338%7C1709519232
```
