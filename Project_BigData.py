#!/usr/bin/env python
# coding: utf-8

# In[89]:


from pyspark.sql import functions as F
from pyspark.sql import DataFrameNaFunctions as DFna
from pyspark.sql.functions import udf, col, when
import matplotlib.pyplot as plt
import pyspark as ps
import os, sys, requests, json
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql import Row
import numpy as np
import math
import pandas as pd
from pandas import Series, DataFrame


# In[90]:


spark = ps.sql.SparkSession.builder             .master("local[4]")             .appName("building recommender")             .getOrCreate()

sc = spark.sparkContext 


# In[36]:


# read movies CSV
movies = spark.read.option("header", "true").csv("/Users/vkoushikmuthyapu/desktop/ml-latest-small/movies.csv",inferSchema=True)
movies.printSchema()
movies.show()


# In[37]:


ratings = spark.read.option("header", "true").csv("/Users/vkoushikmuthyapu/desktop/ml-latest-small/ratings.csv",inferSchema=True)
ratings.printSchema()
ratings.show()


# In[38]:


newrating = ratings.select(['userId', 'movieId', 'rating'])
newrating.show()


# In[39]:


newerratings = newrating.rdd
newerratings


# In[40]:


training_df, validation_df, test_df = newrating.randomSplit([.6, .2, .2], seed=0)
#training_RDD = training_df.rdd.map(lambda x: (x[0], x[1])).cache()
#validation_for_predict_RDD = validation_df.rdd
#test_for_predict_RDD = test_df.rdd.map(lambda x: (x[0], x[1])).cache()
training_df


# In[132]:


als = ALS(maxIter=10, regParam=0.05, rank=18, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", nonnegative= True)          # regularization param)
model = als.fit(training_df)
# make prediction
predictions = model.transform(validation_df)
new_predictions = predictions.filter(col('prediction') != np.nan)
rmse = evaluator.evaluate(new_predictions)
print ("For rank =",18, "reg =", 0.05 ," the RMSE= " ,rmse)


# In[41]:


iterations = 10
regularization_parameter = [0.001, 0.01, 0.05, 0.1, 0.2]
ranks = [8, 10, 12, 14, 16, 18, 20]
errors = []
err = 0
#tolerance = 0.02


# In[80]:


min_error = float('inf')
best_rank = -1
best_iteration = -1

for rank in ranks:
    for reg in regularization_parameter:
        # train ALS model
        als = ALS(maxIter=iterations, regParam=reg, rank=rank, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", nonnegative= True)          # regularization param)
        #model = als.fit(training_df)
        # make prediction
        #predictions = model.transform(validation_df)
        #new_predictions = predictions.filter(col('prediction') != np.nan)
        param_grid= ParamGridBuilder().addGrid(als.rank,[rank]).addGrid(als.maxIter,[10]).addGrid(als.regParam,[reg]).build()
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=5)
        cvModel = crossval.fit(training_df)
        cvModel_pred = cvModel.transform(validation_df)
        cvModel_pred = cvModel_pred.filter(col('prediction') != np.nan)
        rmse = evaluator.evaluate(cvModel_pred)
        errors.append(rmse)
        
            
        print ("For rank =",rank, "reg =", reg ," the RMSE= " ,rmse)
        if rmse < min_error:
            min_error = rmse
            best_rank = rank
            best_reg = reg
print ("The best model was trained with rank= ", best_rank, "With reg= ", best_reg)


# In[82]:


best = cvModel.bestModel


# In[102]:


#here testing with new test data
als = ALS(maxIter=iterations, regParam=0.2, rank=10, userCol="userId", itemCol="movieId", ratingCol="rating")
#param_grid= ParamGridBuilder().addGrid(als.rank,[10]).addGrid(als.maxIter,[10]).addGrid(als.regParam,[0.2]).build()
cvModel = als.fit(training_df)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
cvModel_pred = cvModel.transform(test_df)
cvModel_pred = cvModel_pred.filter(col('prediction') != np.nan)
rmseT = evaluator.evaluate(cvModel_pred)
print("test data Rmse= ", rmseT)
#display(cvModel_pred.sort("userID","rattings"))


# In[103]:


prediction = (cvModel_pred.sort(newrating["userID"]))
prediction.show()


# In[104]:


cvModel_pred = cvModel_pred.na.drop()
cvModel_pred.describe().show()


# In[125]:


user_recs = best.recommendForAllUsers(10)
user_recs


# In[126]:


def recs_users(recs):
    recs = recs.select("recommendations.movieId", "recommendations.rating")
    movies = recs.select("movieId").toPandas().iloc[0,0]
    ratings = recs.select("rating").toPandas().iloc[0,0]
    ratings_matrix = pd.DataFrame(movies, columns = ["movieId"])
    ratings_matrix["ratings"] = ratings
    ratings_matrix_ps = ratings_matrix
    return ratings_matrix_ps


# In[127]:


test = recs_users(user_recs)
test.join(movies)


# In[135]:


cvModel_pred.toPandas()['rating'].hist()
plt.xlabel('ratings')
plt.ylabel('users')
plt.title('rating vs users')
plt.show()


# In[136]:


cvModel_pred.toPandas()['prediction'].hist()
plt.xlabel('predicted_ratings')
plt.ylabel('users')
plt.title('predicted_ratings vs users')
plt.show()





