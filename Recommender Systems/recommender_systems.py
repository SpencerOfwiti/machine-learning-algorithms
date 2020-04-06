# import libraries
import lenskit.datasets as ds
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
import pandas as pd
import csv

# load dataset
data = ds.MovieLens('lab4-recommender-systems/')

# numbers of rows of data to be printed
rows_to_show = 10  # CHANGEME
# minimum number of people to have rated movie in ratings
minimum_to_include = 20  # CHANGEME


# get recommendation according to genre
def get_recommendation(genre):
    # get highest rated movies
    average_ratings = data.ratings.groupby(['item']).mean()
    ratings_counts = data.ratings.groupby(['item']).count()
    average_ratings = average_ratings.loc[ratings_counts['rating'] > minimum_to_include]
    # adding genres
    average_ratings = average_ratings.join(data.movies['genres'], on='item')

    if genre:
        average_ratings = average_ratings.loc[average_ratings['genres'].str.contains(genre)]

    sorted_avg_ratings = average_ratings.sort_values(by='rating', ascending=False)
    # adding titles
    joined_data = sorted_avg_ratings.join(data.movies['title'], on='item')
    joined_data = joined_data[joined_data.columns[3:]]
    return joined_data


# display first few rows of the data
general = get_recommendation(None)
print('RECOMMENDED FOR ANYBODY:\n', general.head(rows_to_show))

action = get_recommendation('Action')
print('RECOMMENDED FOR AN ACTION MOVIE FAN:\n', action.head(rows_to_show))

romance = get_recommendation('Romance')
print('RECOMMENDED FOR AN ROMANCE MOVIE FAN:\n', romance.head(rows_to_show))

jabril_rating_dict = {}
jgb_rating_dict = {}

# getting personalized rating data
with open('lab4-recommender-systems/jabril-movie-ratings.csv', newline='') as csvfile:
    ratings_reader = csv.DictReader(csvfile)
    for row in ratings_reader:
        if (row['ratings'] != '') and (float(row['ratings']) > 0) and (float(row['ratings']) < 6):
            jabril_rating_dict.update({int(row['item']): float(row['ratings'])})

with open('lab4-recommender-systems/jgb-movie-ratings.csv', newline='') as csvfile:
    ratings_reader = csv.DictReader(csvfile)
    for row in ratings_reader:
        if (row['ratings'] != '') and (float(row['ratings']) > 0) and (float(row['ratings']) < 6):
            jgb_rating_dict.update({int(row['item']): float(row['ratings'])})

# number of recommendations to get
num_recs = 10  # CHANGEME

user_user = UserUser(20, min_nbrs=5)  # set minimum and maximum number of neighbours
model = Recommender.adapt(user_user)
model.fit(data.ratings)


# get recommendation according to user
def get_user_recommendation(user_dict):
    recs = model.recommend(-1, num_recs, ratings=pd.Series(user_dict))  # -1 tells us its not an existing user
    joined_data = recs.join(data.movies['genres'], on='item')
    joined_data = joined_data.join(data.movies['title'], on='item')
    joined_data = joined_data[joined_data.columns[2:]]
    return joined_data


jabril_recs = get_user_recommendation(jabril_rating_dict)
print('\nRECOMMENDED FOR JABRIL:', jabril_recs)

jgb_recs = get_user_recommendation(jgb_rating_dict)
print('\nRECOMMENDED FOR JOHN-GREEN-BOT:', jgb_recs)

# creating a combined ratings dictionary
combined_rating_dict = {}
for k in jabril_rating_dict:
    if k in jgb_rating_dict:
        combined_rating_dict.update({k: float((jabril_rating_dict[k]+jgb_rating_dict[k])/2)})
    else:
        combined_rating_dict.update({k: jabril_rating_dict[k]})
for k in jgb_rating_dict:
    if k not in combined_rating_dict:
        combined_rating_dict.update({k: jgb_rating_dict[k]})

print('\nSanity check:')
print("\tJabril's rating for 1197 (The Princess Bride) is", str(jabril_rating_dict[1197]))
print("\tJohn Green Bot's rating for 1197 (The Princess Bride) is", str(jgb_rating_dict[1197]))
print('\tCombined rating for 1197 (The Princess Bride) is', str(combined_rating_dict[1197]))

# hybrid movie recommendations
hybrid_recs = get_user_recommendation(combined_rating_dict)
print('\nRECOMMENDED FOR JABRIL / JOHN-GREEN-BOT HYBRID:', hybrid_recs)
