# Section : Load all libraries needed to carry out analysis  ---------------
library(tidyverse)
library(stringr)
library(caret)
library(matrixStats) 
library(dslabs)
library(rpart)
library(ModelMetrics)
library(lubridate)

# Note: Sections can be collapsed - It makes for easier navigating!

# Section : Load data from file instead of downloading it each time  ---------------
if (!exists("edx") | !exists("validation")) {
  if (file.exists("MovieLensDataSet.RData")) {
    print("Loading dataset from file instead of downloading again")
    Sys.sleep(6)
    load("MovieLensDataSet.RData") 
  } 
  else {
    ################################
    # Create edx set, validation set
    ################################
    
    # Note: this process could take a couple of minutes
    
    if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
    if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
    if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
    
    # MovieLens 10M dataset:
    # https://grouplens.org/datasets/movielens/10m/
    # http://files.grouplens.org/datasets/movielens/ml-10m.zip
    
    dl <- tempfile()
    download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
    
    ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                     col.names = c("userId", "movieId", "rating", "timestamp"))
    
    movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
    colnames(movies) <- c("movieId", "title", "genres")
    movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                               title = as.character(title),
                                               genres = as.character(genres))
    
    movielens <- left_join(ratings, movies, by = "movieId")
    
    # Validation set will be 10% of MovieLens data
    
    set.seed(1, sample.kind="Rounding") 
    test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
    edx <- movielens[-test_index,]
    temp <- movielens[test_index,]
    
    # Make sure userId and movieId in validation set are also in edx set
    
    validation <- temp %>% 
      semi_join(edx, by = "movieId") %>%
      semi_join(edx, by = "userId")
    
    # Add rows removed from validation set back into edx set
    
    removed <- anti_join(temp, validation)
    edx <- rbind(edx, removed)
    
    rm(dl, ratings, movies, test_index, temp, movielens, removed)
  }
    
}


# Section : Data prep and cleaning ---------------
genrelevels <- c("Action", "Adventure", "Animation", "Children", "Comedy", 
                 "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", 
                 "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", 
                 "War", "Western", "(no genres listed)")

## Sub-section : One hot encoding genres and adding release date related variables ---------------
OneHotGenres <- function(dat) {
  # Separates genres into columns filled with 1's and 0's - If a movie is in a given genre
  # the entry is a 1, otherwise 0
  oneHotGenres <- data.frame(do.call(rbind, lapply(dat$genres, function(x) 
    {as.vector(table(lapply(str_split(x, "\\|"), factor, levels = genrelevels)))} )))
  names(oneHotGenres) <- genrelevels
  # Appends one-hot columns to the provided matrix 
  dat <- bind_cols(dat, oneHotGenres) 
  dat
}

# Encoding genres and extracting releaseYear takes a while, this next set of IF statements 
# Only encodes values and extracts releaseYear values if it hasn't happened yet, otherwise
# it will load it from the saved data file.
if (!exists("edxOneHot")) {
  if (!file.exists("MovieLens_EdxOneHot.RData")) {
    edxOneHot <- OneHotGenres(edx)
    ## Sub-section : Adding releaseYear column and related ratings/year ---------------
    edxOneHot <- edxOneHot %>%
      group_by(movieId) %>% 
      mutate(releaseYear = as.numeric(str_extract(str_extract(title[1], "\\(\\d{4,}\\)"), "\\d{4,}")),
             movieRatingsCount = n(),
             movieRatingsPerYear = movieRatingsCount / as.numeric(interval(as_datetime(paste(releaseYear, 1, 1)), 
                                                                           as_datetime(paste(2010, 1, 10))) / years(1))) %>%
      ungroup() 
    save(edxOneHot, file = "MovieLens_EdxOneHot.RData")
  } else {
    load("MovieLens_EdxOneHot.RData")
  }
}


# Section: Split edxOneHot into train and test sets to determine optimal lambda values ------------
# Train/Test split and lambda testing similar to that used in the course text
shouldTestLambdas <- TRUE
if(shouldTestLambdas) {
  test_index <- createDataPartition(y = edxOneHot$rating, times = 1, p = 0.3, list = FALSE)
  train_set <- edxOneHot[-test_index,]
  test_set <- edxOneHot[test_index,] 
  
  lambdas <- seq(2, 6, 0.25)
  
  mu <- mean(train_set$rating)
  just_the_sum <- train_set %>% 
    group_by(movieId) %>% 
    summarize(s = sum(rating - mu), n_i = n())
  
  rmsesMovies <- sapply(lambdas, function(l){
    predicted_ratings <- test_set %>% 
      left_join(just_the_sum, by='movieId') %>% 
      mutate(b_i = s/(n_i+l)) %>%
      mutate(pred = mu + b_i) %>%
      pull(pred) 
    return(RMSE(predicted_ratings, test_set$rating, na.rm = TRUE))
  })
  qplot(lambdas, rmsesMovies)   
  # For a 70/30 train/test split, a lambda = 2.5 appears to be optimal for the Movie Weight alone
  
  # Same idea, but for determining the lambda for the movie + user weights ------------
  lambdas <- seq(2, 6, 0.25)
  
  rmsesMovUsers <- sapply(lambdas, function(l){
    
    mu <- mean(train_set$rating)
    
    b_i <- train_set %>% 
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l))
    
    b_u <- train_set %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+l))
    
    predicted_ratings <- 
      test_set %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_i + b_u) %>%
      pull(pred)
    
    return(RMSE(predicted_ratings, test_set$rating, na.rm = TRUE))
  })
  
  qplot(lambdas, rmsesMovUsers)  
  # For a 70/30 train/test split, a lambda = 5.0 appears to be optimal for the Movie + User weights
  
  # Same idea again, but for determining the lambda for the movie + user + year weights ------------
  lambdas <- seq(2, 6, 0.25)
  
  rmsesMovUsersYear <- sapply(lambdas, function(l){
    print(l)
    mu <- mean(train_set$rating)
    
    b_i <- train_set %>% 
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l))
    
    b_u <- train_set %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+l))
    
    b_y <- train_set %>% 
      left_join(b_i, by="movieId") %>%
      left_join(b_u, by="userId") %>%
      group_by(releaseYear) %>%
      summarize(b_y = sum(rating - b_i - b_u - mu)/(n()+l))
    
    predicted_ratings <- 
      test_set %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      left_join(b_y, by = "releaseYear") %>%
      mutate(pred = mu + b_i + b_u + b_y) %>%
      pull(pred)
    
    return(RMSE(predicted_ratings, test_set$rating, na.rm = TRUE))
  })
  
  qplot(lambdas, rmsesMovUsersYear)  
  # For a 70/30 train/test split, a lambda = 4.75 appears to be optimal for the Movie + User + Year weights
}



# Section: Create needed dataframes for prediction using optimal lambda value ------------
mu <- mean(edxOneHot$rating)
lambda = 4.75

# First calculate the movie weights
weightsMovies <- edxOneHot %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda),
            movieReviews = n() )  

# Then the user weights
weightsUsers <- edxOneHot %>% 
  left_join(weightsMovies, by="movieId") %>% 
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda),
            userReviews = n() )

# Then the year weights
weightsYears <- edxOneHot %>% 
  left_join(weightsMovies, by="movieId") %>% 
  left_join(weightsUsers, by="userId") %>% 
  group_by(releaseYear) %>%
  summarize(yearMovieReleases = n_distinct(movieId), 
            yearAverage = mean(rating),
            yearAverageMod = mean(rating - b_i - b_u - mu),
            b_y = sum(rating - b_i - b_u - mu)/(n() + lambda))

# Then combine all of the weights into a single dataframe and calculate the 
# modified rating - avg, movie, user, year effects removed
edxWeighted <- edxOneHot %>%
  left_join(weightsMovies, by="movieId") %>% 
  left_join(weightsUsers, by="userId") %>% 
  left_join(weightsYears, by="releaseYear") %>% 
  mutate(modRating = rating - b_i - b_u - b_y - mu)  


# Section: Create Separated Genre Weights -------------
# Create a dataframe with only the one-hot encoded values
genreStats <- edxWeighted %>%
  select(7:24)
# Fit the modrating values onto the one-hot df
genreRatings <- (genreStats * edxWeighted$modRating )
# Replace 0's with NA so we can calculate related values for actual movie reviews - not empty data
genreRatings[genreRatings == 0] <- NA
# Sum the ratings for each genre
genreRatings <- genreRatings %>%
  summarize_all(list(sum), na.rm = TRUE)

# Select distinct movies and their genres
genreWeights_Separate <- edxWeighted %>%
  select(movieId, 7:24) %>%
  distinct()

# Calculate the weights per genre
for (i in names(genreRatings)) {
  genreRatings[[i]] <- (genreRatings[[i]]) / (lambda + sum(genreStats[[i]]))
  genreWeights_Separate[[i]] <- genreRatings[[i]] * genreWeights_Separate[[i]]
}
# Create the weights for separated genres
genreWeights_Separate <- genreWeights_Separate %>%
  mutate(b_g_s = rowSums(select(., -movieId) )) %>%
  select(movieId, b_g_s)


# Section: Create Combined Genre Weight -------------
genreWeights_Combined <- edxWeighted %>%
  group_by(genres) %>%
  summarize(b_g_c = sum(modRating) / (lambda + n()))
 

# Section: Create Movie Popularity Weight -------------
# The movie popularity weight did not prove useful and so it was removed from this analysis and accompanying discussion

# moviePopularityPrefWeights <- edxWeighted %>%
#   left_join(genreWeights_Combined, by = "genres") %>%
#   mutate(modRating = rating - b_i - b_u - b_y - b_g_c - mu)  %>%
#   group_by(userId) %>%
#   summarize(meanModRating =  mean(modRating),
#             moviePopPref = sum(modRating * movieReviews) / (sum(movieReviews) + lambda) )  
 

# Section: Testing of all weighted approaches on validation set -------------
# First, mutate the releaseYear variable and then join all weight tables based
# on movieId, userId, releaseYear, or genres
rsmeResults <- validation %>%
  group_by(movieId) %>% 
  mutate(releaseYear = as.numeric(str_extract(str_extract(title[1], "\\(\\d{4,}\\)"), "\\d{4,}"))) %>%
  ungroup() %>% 
  left_join(weightsMovies, by="movieId") %>% 
  left_join(weightsUsers, by="userId")  %>%  
  left_join(weightsYears, by="releaseYear") %>%
  left_join(genreWeights_Combined, by="genres") %>%
  left_join(genreWeights_Separate, by = "movieId") %>%
  #left_join(moviePopularityPrefWeights , by = "userId")  %>%  # Popularity measure was removed
  summarize(`Mean Rating` = RMSE(rep(mu, nrow(.)), rating),
            `Movie Weights` = RMSE(mu + b_i, rating), 
            `Movie + User Weights` = RMSE(mu + b_i + b_u, rating), 
            `Movie + User + Year Weights` = RMSE(mu + b_i + b_u + b_y, rating), 
            `Movie + User + Year + Genre Weights (Separated Genres)` = RMSE(mu + b_i + b_u + b_y + b_g_s, rating), 
            `Movie + User + Year + Genre Weights (Combined Genres)` = RMSE(mu + b_i + b_u + b_y + b_g_c, rating),
            #`All weights  + Movie Popularity` = RMSE(mu + b_i + b_u + b_y + b_g_c + moviePopPrefNew, rating)# Popularity measure was removed
            )
rownames(rsmeResults) <- c("RMSE")

# Section: Display Final Results!
print(t(rsmeResults))



