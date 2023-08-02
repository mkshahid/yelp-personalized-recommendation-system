# yelp-personalized-recommendation-system

# Method Description:
This recommender system is model-based, which uses engineered features to build the train and test inputs for the XGBoost model. The model uses data that is collected from business.json, checkin.json, photo.json, user.json, and yelp_train.csv. For the business.json file, the average rating, number of reviews, price range, open status, noise level (using one-hot encoding), number of attributes, number of true attributes, number of categories, and attire (using one-hot encoding) is collected for each business in that json file. For the checkin.json file, the total number of check-ins is collected for each business in that file. For the photo.json file, the total number of photos for each business is collected from that file. For the user.json file, each user's average rating, number of reviews, num useful/fans/funny/cool, and number of days since joining Yelp is collected. I spent a lot of time tuning the hyperparameters to find a balance between accuracy, speed, and fit (i.e., not overfitting or underfitting). I also used GridSearchCV to help me determine which values for my hyperparameters would be ideal. I also considered using my item-based CF, but it only worsened my results every time. Also, for the missing features from my validation set, I generally used averages to fill in those unknowns.

# Error Distribution:
>=0 and <1: 61015
>=1 and <2: 17493
>=2 and <3: 893
>=3 and <4: 39
>=4: 1

# RMSE:
0.9793504352906133

# Execution Time:
271.2546498775482
