## Method Description:
# This recommender system is model-based, which uses engineered features to build the train and test inputs for the
# XGBoost model. The model uses data that is collected from business.json, checkin.json, photo.json, user.json, and
# yelp_train.csv. For the business.json file, the average rating, number of reviews, price range, open status, noise
# level (using one-hot encoding), number of attributes, number of true attributes, number of categories, and attire
# (using one-hot encoding) is collected for each business in that json file. For the checkin.json file, the total
# number of checkins is collected for each business in that file. For the photo.json file, the total number of photos
# for each business is collected from that file. For the user.json file, each user's average rating, number of reviews,
# num useful/fans/funny/cool, and number of days since joining Yelp is collected. I spent a lot of time tuning the
# hyperparameters to find a balance between accuracy, speed, and fit (i.e., not overfitting or underfitting). I also
# used GridSearchCV to help me determine which values for my hyperparameters would be ideal. I considered also using
# my item-based CF; however, using it only worsened my results everytime. Also, for my features that were missing for
# my validation set, I generally used means to fill in those unknowns.

## Error Distribution:
# >=0 and <1: 61015
# >=1 and <2: 17493
# >=2 and <3: 893
# >=3 and <4: 39
# >=4: 1

## RMSE:
# 0.9793504352906133

## Execution Time:
# 271.2546498775482

from datetime import datetime
from pyspark import SparkContext
import pandas as pd
import sys
import xgboost as xgb
import json
import time

start_time = time.time()


def attireToNum(attire_str):
    if attire_str == "casual":
        return 1
    elif attire_str == "formal":
        return 3
    elif attire_str == "dressy":
        return 2
    else:
        return 0


def isAttireCasual(attire_str):
    if attire_str == "casual":
        return 1
    else:
        return 0


def isAttireFormal(attire_str):
    if attire_str == "formal":
        return 1
    else:
        return 0


def isAttireDressy(attire_str):
    if attire_str == "dressy":
        return 1
    else:
        return 0


def noiseLevelIsQuiet(noise_level_str):
    if noise_level_str == "quiet":
        return 1
    else:
        return 0


def noiseLevelIsAverage(noise_level_str):
    if noise_level_str == "average":
        return 1
    else:
        return 0


def noiseLevelIsLoud(noise_level_str):
    if noise_level_str == "loud":
        return 1
    else:
        return 0


def noiseLevelIsVeryLoud(noise_level_str):
    if noise_level_str == "very_loud":
        return 1
    else:
        return 0


def get_num_true_attributes(data):
    if not data:
        return 0

    true_count = 0
    for value in data.values():
        if value == 'True':
            true_count += 1
        elif isinstance(value, dict):
            true_count += get_num_true_attributes(value)

    return true_count


folder_path = sys.argv[1]
# folder_path = "../resource/asnlib/publicdata"
test_file_name = sys.argv[2]
# test_file_name = "../resource/asnlib/publicdata/yelp_val.csv"
output_file_name = sys.argv[3]
# output_file_name = "./output2_2.csv"

sc = SparkContext('local[*]', 'task2_2')

sc.setLogLevel("WARN")

train_data = sc.textFile(folder_path + "/yelp_train.csv").filter(lambda x: "business_id" not in x).map(
    lambda line: line.split(",")).persist()

val_data = sc.textFile(test_file_name).filter(lambda x: "business_id" not in x).map(
    lambda line: line.split(",")).persist()

user_data = sc.textFile(folder_path + "/user.json").map(lambda x: json.loads(x))

avg_stars_user = user_data.map(lambda x: float(x["average_stars"])).mean()
avg_num_reviews_user = user_data.map(lambda x: float(x["review_count"])).mean()
avg_num_useful_user = user_data.map(lambda x: float(x["useful"])).mean()
avg_num_fans = user_data.map(lambda x: float(x["fans"])).mean()
avg_num_funny = user_data.map(lambda x: float(x["funny"])).mean()
avg_num_cool = user_data.map(lambda x: float(x["cool"])).mean()
avg_num_days_since_joining = user_data.map(
    lambda x: (datetime.now() - datetime.strptime(x["yelping_since"], "%Y-%m-%d")).total_seconds() / 86400).mean()

user_data = user_data.map(
    lambda x: [x["user_id"], float(x["average_stars"]), int(x["review_count"]), int(x["useful"]), int(x["fans"]),
               int(x["funny"]), int(x["cool"]),
               (datetime.now() - datetime.strptime(x["yelping_since"], "%Y-%m-%d")).total_seconds() / 86400]) \
    .persist()

checkin_data = sc.textFile(folder_path + '/checkin.json').map(lambda x: json.loads(x)).map(
    lambda x: [x['business_id'], sum(x['time'].values())])

photo_data = sc.textFile(folder_path + '/photo.json').map(lambda x: json.loads(x)) \
    .map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y).mapValues(lambda x: x)

business_data = sc.textFile(folder_path + "/business.json").map(lambda x: json.loads(x))
avg_stars_business = business_data.map(lambda x: float(x["stars"])).mean()
avg_num_reviews_business = business_data.map(lambda x: float(x["review_count"])).mean()
avg_num_attributes = business_data.map(
    lambda x: len(x['attributes'].keys()) if x['attributes'] is not None else 0).mean()
avg_num_true_attributes = business_data.map(lambda x: get_num_true_attributes(x['attributes'])).mean()
avg_num_categories = business_data.map(lambda x: len(x['categories']) if x['categories'] is not None else 0).mean()
business_data = business_data.map(lambda x: [x["business_id"], float(x["stars"]), int(x["review_count"]),
                                             int(x['attributes']['RestaurantsPriceRange2']) if x[
                                                                                                   'attributes'] is not None and 'attributes' in x and 'RestaurantsPriceRange2' in
                                                                                               x['attributes'] else 2,
                                             int(x["is_open"]), noiseLevelIsQuiet(x['attributes']['NoiseLevel']) if x[
                                                                                                                        'attributes'] is not None and 'attributes' in x and 'NoiseLevel' in
                                                                                                                    x[
                                                                                                                        'attributes'] else 0,
                                             noiseLevelIsAverage(x['attributes']['NoiseLevel']) if x[
                                                                                                       'attributes'] is not None and 'attributes' in x and 'NoiseLevel' in
                                                                                                   x[
                                                                                                       'attributes'] else 0,
                                             noiseLevelIsLoud(x['attributes']['NoiseLevel']) if x[
                                                                                                    'attributes'] is not None and 'attributes' in x and 'NoiseLevel' in
                                                                                                x['attributes'] else 0,
                                             noiseLevelIsVeryLoud(x['attributes']['NoiseLevel']) if x[
                                                                                                        'attributes'] is not None and 'attributes' in x and 'NoiseLevel' in
                                                                                                    x[
                                                                                                        'attributes'] else 0,
                                             len(x['attributes']) if x['attributes'] is not None else 0,
                                             get_num_true_attributes(x['attributes']),
                                             len(x['categories']) if x['categories'] is not None else 0,
                                             isAttireCasual(x['attributes']['RestaurantsAttire']) if x[
                                                                                                         'attributes'] is not None and 'attributes' in x and 'RestaurantsAttire' in
                                                                                                     x[
                                                                                                         'attributes'] else 0,
                                             isAttireDressy(x['attributes']['RestaurantsAttire']) if x[
                                                                                                         'attributes'] is not None and 'attributes' in x and 'RestaurantsAttire' in
                                                                                                     x[
                                                                                                         'attributes'] else 0,
                                             isAttireFormal(x['attributes']['RestaurantsAttire']) if x[
                                                                                                         'attributes'] is not None and 'attributes' in x and 'RestaurantsAttire' in
                                                                                                     x[
                                                                                                         'attributes'] else 0]) \
    .persist()
business_data = business_data.map(lambda x: (x[0], x[1:])).leftOuterJoin(
    checkin_data.map(lambda x: (x[0], x[1:]))).leftOuterJoin(photo_data.map(lambda x: (x[0], x[1:])))

business_data = business_data.map(
    lambda x: [x[0], x[1][0][0][0], x[1][0][0][1], x[1][0][0][2], x[1][0][0][3], x[1][0][0][4], x[1][0][0][5],
               x[1][0][0][6], x[1][0][0][7], x[1][0][0][8], x[1][0][0][9], x[1][0][0][10], x[1][0][0][11],
               x[1][0][0][12], x[1][0][0][13], x[1][0][1][0] if x[1][0][1] != None else 0,
               x[1][1][0] if x[1][1] != None else 0])

# Split train_data into (user_id, business_id, stars)
train_split = train_data.map(lambda x: (x[0], x[1], float(x[2])))

# Join train_split with user_data based on user_id
train_user = train_split.map(lambda x: (x[0], (x[1], x[2]))).join(user_data.map(lambda x: (x[0], x)))
# print(train_user.collect())

# Join train_user with business_data based on business_id
train_user_business = train_user.map(lambda x: (x[1][0][0], (x[1][0][1], x[1][1]))).join(
    business_data.map(lambda x: (x[0], x)))
# Transform the RDD into a list of lists with the desired fields # stars, avg stars (user), # reviews (user), stars (biz), # reviews (biz)
train = train_user_business.map(
    lambda x: [x[1][0][0], x[1][0][1][1], x[1][0][1][2], x[1][0][1][3], x[1][0][1][4], x[1][1][1], x[1][1][2],
               x[1][1][3], x[1][1][4], x[1][1][5], x[1][1][6], x[1][1][7], x[1][1][8], x[1][1][9], x[1][1][10],
               x[1][1][11], x[1][1][12], x[1][1][13], x[1][1][14], x[1][1][15], x[1][1][16], x[1][0][1][5],
               x[1][0][1][6], x[1][0][1][7]])

X_train = train.map(lambda x: x[1:])
X_train = pd.DataFrame(X_train.collect(),
                       columns=['avg_stars_user', 'num_reviews_user', 'num_useful_user', 'num_fans_user', 'stars_biz',
                                'num_reviews_biz', 'pricing_biz', 'is_open', 'is_quiet', 'is_average',
                                'is_loud', 'is_very_loud', 'num_attributes', 'num_true_attributes', 'num_categories',
                                'is_casual', 'is_dressy', 'is_formal', 'num_checkin', 'num_photos', 'num_funny_user',
                                'num_cool_user', 'days_since_joining'])
y_train = train.map(lambda x: x[0])
y_train = pd.DataFrame(y_train.collect(), columns=['stars'])
#

# Split val_data into (user_id, business_id)
val_split = val_data.map(lambda x: (x[0], x[1]))

# Join train_split with user_data based on user_id
val_user = val_split.map(lambda x: (x[0], (x[1]))).join(user_data.map(lambda x: (x[0], x)))

# Join train_user with business_data based on business_id
val_user_business = val_user.map(lambda x: (x[1][0], (x[1][1]))).join(
    business_data.map(lambda x: (x[0], x)))

# Transform the RDD into a list of lists with the desired fields # stars, avg stars (user), # reviews (user), stars (biz), # reviews (biz)
val = val_user_business.map(lambda x: [
    x[1][0][1] if x[1][0][1] is not None else avg_stars_user,
    x[1][0][2] if x[1][0][2] is not None else avg_num_reviews_user,
    x[1][0][3] if x[1][0][3] is not None else avg_num_useful_user,
    x[1][0][4] if x[1][0][4] is not None else avg_num_fans,
    x[1][1][1] if x[1][1][1] is not None else avg_stars_business,
    x[1][1][2] if x[1][1][2] is not None else avg_num_reviews_business,
    x[1][1][3],
    x[1][1][4],
    x[1][1][5],
    x[1][1][6],
    x[1][1][7],
    x[1][1][8],
    x[1][1][9],
    x[1][1][10] if x[1][0][5] is not None else avg_num_attributes,
    x[1][1][11] if x[1][0][5] is not None else avg_num_true_attributes,
    x[1][1][12] if x[1][0][5] is not None else avg_num_categories,
    x[1][1][13],
    x[1][1][14],
    x[1][1][15],
    x[1][1][16],
    x[1][0][5] if x[1][0][5] is not None else avg_num_funny,
    x[1][0][6] if x[1][0][6] is not None else avg_num_cool,
    x[1][0][7] if x[1][0][7] is not None else avg_num_days_since_joining])

# X_val = val.map(lambda x: x[1:])
X_val = pd.DataFrame(val.collect(),
                     columns=['avg_stars_user', 'num_reviews_user', 'num_useful_user', 'num_fans_user', 'stars_biz',
                              'num_reviews_biz', 'pricing_biz', 'is_open', 'is_quiet', 'is_average',
                              'is_loud', 'is_very_loud', 'num_attributes', 'num_true_attributes', 'num_categories',
                              'is_casual', 'is_dressy', 'is_formal', 'num_checkin', 'num_photos', 'num_funny_user',
                              'num_cool_user', 'days_since_joining'])
# y_val = val.map(lambda x: float(x[0]))
# y_val = pd.DataFrame(y_val.collect(), columns=['stars'])

# xgb = xgb.XGBRegressor(objective='reg:squarederror')
#
# from sklearn.model_selection import GridSearchCV
# # Define hyperparameters grid
# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [.05, 0.1, .15]
# }
#
# # Define GridSearchCV object
# grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
#
# grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
#
# # Print best hyperparameters and best score
# print("Best hyperparameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)

# cb = catboost.CatBoostRegressor(eta=0.07, max_depth=8, subsample=0.875, colsample_bylevel=0.7, num_boost_round=275,
#                        l1_leaf_reg=0.01, l2_leaf_reg=0.01)
#
# cb.fit(X_train, y_train)
#
# y_val_pred = cb.predict(X_val)

xgb = xgb.XGBRegressor(learning_rate=0.07, max_depth=8, subsample=0.875, colsample_bytree=0.7, n_estimators=275,
                       reg_alpha=0.01, reg_lambda=0.01)
xgb.fit(X_train, y_train)

y_val_pred = xgb.predict(X_val)

users_businesses = val_user_business.map(lambda x: (x[1][0][0], x[0])).collect()

with open(output_file_name, 'w') as file:
    file.write('user_id, business_id, prediction\n')
    for i in range(len(y_val_pred)):
        file.write(str(users_businesses[i][0]) + ',' + str(users_businesses[i][1]) + ',' + str(y_val_pred[i]) + '\n')

# with open("./description.txt", "w") as file:
#     file.write("Method Description:\n")
#     file.write("<enter description here>")
#     file.write("\n\nError Distribution:")
#     file.write("\n>=0 and <1: " + str(len([abs(e) for e in errors if 0 <= e < 1])))
#     file.write("\n>=1 and <2: " + str(len([abs(e) for e in errors if 1 <= e < 2])))
#     file.write("\n>=2 and <3: " + str(len([abs(e) for e in errors if 2 <= e < 3])))
#     file.write("\n>=3 and <4: " + str(len([abs(e) for e in errors if 3 <= e < 4])))
#     file.write("\n>=4: " + str(len([abs(e) for e in errors if e >= 4])))
#     file.write("\n\nRMSE:\n")
#     file.write(str(rmse))
#     file.write("\n\nExecution Time:\n")
#     file.write(str(time.time() - start_time))
#
# print(time.time() - start_time)
