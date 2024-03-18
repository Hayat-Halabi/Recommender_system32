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
user_util_matrix.T
user_util_matrix = util_mat.copy()

# Filling the NaN values in the utility matrix rows with the corresponding user's mean ratings.
# This step is necessary for calculating Pearson correlation, and it assumes average ratings for unrated movies.
user_util_matrix = user_util_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)

# Displaying the first 5 rows of the modified utility matrix.
user_util_matrix.head(5)

# Transposing the user_util_matrix to get movies as rows and users as columns,
# then calculating the Pearson correlation matrix between movies.
user_util_matrix.T.corr()
# Creating a copy of the utility matrix to work with.
user_util_matrix = util_mat.copy()

# Filling the NaN values in the utility matrix rows with the corresponding user's mean ratings.
# This step is necessary for calculating Pearson correlation, and it assumes average ratings for unrated movies.
user_util_matrix = user_util_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)

# Displaying the first 5 rows of the modified utility matrix.
user_util_matrix.head(5)

# Transposing the user_util_matrix to get movies as rows and users as columns,
# then calculating the Pearson correlation matrix between movies.
user_util_matrix.T.corr()

# Calculating the Pearson correlation matrix between movies using the transposed user_util_matrix.
user_corr_mat = user_util_matrix.T.corr()

# Extracting the correlation values for the first movie (index 0) from the correlation matrix.
corr_user_1 = user_corr_mat.iloc[0]

# Displaying the correlation values for the first movie.
corr_user_1

# Sorting the correlation values for the first movie in descending order.
# This will show the movies that are most correlated (similarly rated) with the first movie.
corr_user_1.sort_values(ascending=False, inplace=True)

# The 'corr_user_1' Series is now sorted in place, with the highest correlations at the top.
corr_user_1

# Dropping NaN values from the 'corr_user_1' Series.
# These NaN values are generated because the standard deviation is zero, which is required for calculating Pearson Similarity.
corr_user_1.dropna(inplace=True)

# The 'corr_user_1' Series is now free of NaN values after dropping them in place.

# Extracting the top 50 correlation values from the 'corr_user_1' Series.
# The first value is neglected as it represents the correlation with itself (user1 itself).
top50_corr_users = corr_user_1[1:51]

# The 'top50_corr_users' Series now contains the top 50 correlation values with other users.

# Filtering the DataFrame 'df_combined' to select rows where 'userId' is equal to 1.
df_combined[df_combined['userId'] == 1]

# The 'filtered_data_user_1' DataFrame now contains all rows where the user ID is 1.

df_combined # user-item matrix

df_combined[(df_combined['movieId'] == 32)]

# Filtering the DataFrame 'df_combined' to select rows where 'userId' is equal to 1 and 'movieId' is equal to 32.
filtered_data_user1_movie32 = df_combined[(df_combined['userId'] == 1) & (df_combined['movieId'] == 32)]
# The 'filtered_data_user1_movie32' DataFrame now contains the rows where user1 has not rated movie with ID 32.

# If user1 has NOT rated movie with ID 32, then filtered_data_user1_movie32 will be empty
filtered_data_user1_movie32
# DataFrame 'df_n_ratings' groups movies by title and calculates the mean rating of each movies and number of ...
# ... ratings each movie received
df_n_ratings
# Mean rating and # of ratings of 32nd movie: 'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
df_n_ratings.loc[['Twelve Monkeys (a.k.a. 12 Monkeys) (1995)']]
# Extracting the user IDs from the 'top50_corr_users'.
top50_users = top50_corr_users.keys()

# Initializing a counter to keep track of the number of users meeting a certain condition.
count = 0

# Creating an empty list to store user IDs.
users = list()

# Iterating through the 'top50_users' list.
for user in top50_users:
    # Get ratings each similar user gave to the 32nd movie
    if df_combined[(df_combined['userId'] == user) & (df_combined['movieId'] == 32)]['rating'].sum():
        # If the condition is met, incrementing the counter and appending the user to the 'users' list.
        count += 1
        users.append(user) # List of users that DID give a rating to 32nd movie

# Printing the final count of users meeting the condition.
print(count) # count has the number of similar user who DID rate the 32nd movie
# Defining a function to predict what user1 will rate the movie using a weighted average of k similar users.
# Weighted ratings
def predict_rating():
    # Initialize variables to calculate weighted average and sum of similarities.
    sum_similarity = 0
    weighted_ratings = 0

    # Loop through each user in the 'users' list.
    for user in users:
        # Calculate the weighted ratings using similarity score and the user's rating for movie ID 32.
        # top50_corr_users.loc[user] = correlation between user 1 and similar users who rated 32nd movie
        # df_combined[(df_combined['userId'] == user) & (df_combined['movieId'] == 32)]['rating'].sum() = rating ...
        # ... the similar user gave to 32nd movie
        weighted_ratings += top50_corr_users.loc[user] * df_combined[(df_combined['userId'] == user) &
                                                                      (df_combined['movieId'] == 32)]['rating'].sum()

        # Accumulate the sum of similarity scores.
        # sum_similarity = sum of correlations between user 1 and similar users
        sum_similarity += top50_corr_users.loc[user]

    # Calculate and print the predicted rating using the weighted average.
    # The predicted rating by user 1
    print(weighted_ratings / sum_similarity)

# Calling the 'predict_rating' function to calculate and print the predicted rating for user1 and movie ID 32.
predict_rating()

df_m[df_m['movieId'] == 32] # movie ID 32
# Creating a copy of the utility matrix to work with for item-based collaborative filtering.
item_util_matrix = util_mat.copy()

# Displaying the first 10 rows of the item_util_matrix.
item_util_matrix.head(10)
# Filling the NaN values in the item_util_matrix columns with the corresponding movie's mean ratings.
# This step is necessary for calculating Pearson correlation, and it assumes average ratings for users that have not rated a movie.
item_util_matrix = item_util_matrix.apply(lambda col: col.fillna(col.mean()), axis=0)

# Displaying the first 5 rows of the item_util_matrix after filling NaN values.
item_util_matrix.head()
# Checking for NaN values in the item_util_matrix using the isna() function, and then summing the total number of NaN values.
nan_count = item_util_matrix.isna().sum()
nan_count
# Calculating the Pearson correlation between movies using the item_util_matrix.
item_corr_matrix = item_util_matrix.corr()

# The 'item_correlation_matrix' now contains the correlation coefficients between different movies.
item_corr_matrix

# Locating rows in the DataFrame 'df_n_ratings' where the movie title is 'Jurassic Park (1993)'.
# The double square brackets are used to ensure the result is a DataFrame, not a Series.
df_n_ratings.loc[['Jurassic Park (1993)']]

# The 'filtered_jurassic_park_ratings' DataFrame now contains the rows corresponding to the movie 'Jurassic Park ###
#(1993)'
# Extracting the correlation values of the movie 'Jurassic Park (1993)' from the item_correlation_matrix.
jurassic_park_corr = item_corr_matrix['Jurassic Park (1993)']

# Sorting the correlation values in descending order.
jurassic_park_corr = jurassic_park_corr.sort_values(ascending=False)
jurassic_park_corr

# Dropping any NaN values from the 'jurassic_park_corr' Series.
jurassic_park_corr.dropna(inplace=True)
jurassic_park_corr

# Extracting the correlation values of the movie 'Jurassic Park (1993)' from the item_correlation_matrix.
jurassic_park_corr = item_corr_matrix['Jurassic Park (1993)']

# Sorting the correlation values in descending order.
jurassic_park_corr = jurassic_park_corr.sort_values(ascending=False)

# Dropping any NaN values from the 'jurassic_park_corr' Series.
jurassic_park_corr.dropna(inplace=True)

# Creating a DataFrame 'movies_similar_to_jurassic_park' with correlation values as data,
# 'Correlation' as column name, and movie titles as index from the 'jurassic_park_corr' Series.
movies_similar_to_jurassic_park = pd.DataFrame(data=jurassic_park_corr.values, columns=['Correlation'],
                                               index=jurassic_park_corr.index)
movies_similar_to_jurassic_park

# Joining the 'total ratings' column from the 'df_n_ratings' DataFrame to 'movies_similar_to_jurassic_park'.

# Get total ratings of movies similar to jurassic park
movies_similar_to_jurassic_park = movies_similar_to_jurassic_park.join(df_n_ratings['total ratings'])
movies_similar_to_jurassic_park

# The 10 movies most similar to the movie Jurassic Park (1993).
movies_similar_to_jurassic_park.head(11)







```
Links to access data
```
movies_csv:
https://vocproxy-1-9.us-west-2.vocareum.com/files/home/labsuser/movies.csv?_xsrf=2%7C23e987d6%7C3998626cb435d6edd51bde567dfcb338%7C1709519232
ratings_csv:
 https://vocproxy-1-9.us-west-2.vocareum.com/files/home/labsuser/ratings.csv?_xsrf=2%7C23e987d6%7C3998626cb435d6edd51bde567dfcb338%7C1709519232
```
