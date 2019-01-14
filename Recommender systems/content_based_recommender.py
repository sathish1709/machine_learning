# library
import pandas as pd

#Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('movies.csv')
#Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('ratings.csv')

print(movies_df.head())

#movies.csv: Actual columns: movieId, title, genres
#Splitting years from title column
#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
#Extracting integer value with paranthesis from title columns
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
print(movies_df['year'].values)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
print(movies_df['year'].values)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df.head()

#Split using pipe delimiter. Now the genre column will have list of all genre
movies_df['genres'] = movies_df.genres.str.split('|')
print(movies_df['genres'].values)

#backup copy
moviesWithGenres_df = movies_df.copy()

#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    #row : returns each row with columns and values.
    #print("Rows ",row)
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1

#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
print(moviesWithGenres_df.head())

print(moviesWithGenres_df["genres"].values)


#create a userInput dataframe with ovie title and user rating.
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ]
inputMovies = pd.DataFrame(userInput)
print(inputMovies.head())

#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
print("Input ID ", inputId.values )
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
print("Input movies", inputMovies.head())
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original
#dataframe or it might spelled differently, please check capitalisation.
print("Input movies", inputMovies.head())


userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
print("Movies with genre ",userMovies.values)
#Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
#Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print("User genre table ", userGenreTable)
print("inputMovies['rating'] ",inputMovies['rating'])

#Dot produt to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
#The user profile
print("User profile ",userProfile)

#Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
#And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
genreTable.head()

#Multiply the genres by the weights and then take the weighted average [genreTable* userProfile]
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
print(recommendationTable_df.head())

#Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
#Just a peek at the values
print(recommendationTable_df.head())

#The final recommendation table
print(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())])