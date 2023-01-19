import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
import predefined
import numpy as np

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

tf.keras.backend.set_floatx('float32')

# Load the dataset
train_df = pd.read_csv("./datasets/california_housing_train.csv")
test_df = pd.read_csv("./datasets/california_housing_test.csv")

# Scale the labels
scale_factor = 1000.0
# Scale the training set's label.
train_df["median_house_value"] /= scale_factor 

# Scale the test set's label
test_df["median_house_value"] /= scale_factor

# Shuffle the examples
train_df = train_df.reindex(np.random.permutation(train_df.index))

# # Create an empty list that will eventually hold all feature columns.
# feature_columns = []

# # Create a numerical feature column to represent latitude.
# latitude = tf.feature_column.numeric_column("latitude")
# feature_columns.append(latitude)

# # Create a numerical feature column to represent longitude.
# longitude = tf.feature_column.numeric_column("longitude")
# feature_columns.append(longitude)

# # Convert the list of feature columns into a layer that will ultimately become
# # part of the model. Understanding layers is not important right now.
# fp_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# resolution_in_degrees = 1.0 

# # Create a new empty list that will eventually hold the generated feature column.
# feature_columns = []

# # Create a bucket feature column for latitude.
# latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
# latitude_boundaries = list(np.arange(int(min(train_df['latitude'])), 
#                                      int(max(train_df['latitude'])), 
#                                      resolution_in_degrees))
# latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, 
#                                                latitude_boundaries)
# feature_columns.append(latitude)

# # Create a bucket feature column for longitude.
# longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
# longitude_boundaries = list(np.arange(int(min(train_df['longitude'])), 
#                                       int(max(train_df['longitude'])), 
#                                       resolution_in_degrees))
# longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column, 
#                                                 longitude_boundaries)
# feature_columns.append(longitude)

resolution_in_degrees = 0.4

# Create a new empty list that will eventually hold the generated feature column.
feature_columns = []

# Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df['latitude'])), int(max(train_df['latitude'])), resolution_in_degrees))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, latitude_boundaries)

# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df['longitude'])), int(max(train_df['longitude'])), resolution_in_degrees))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column, longitude_boundaries)

# Create a feature cross of latitude and longitude.
latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

# Convert the list of feature columns into a layer that will later be fed into
# the model. 
feature_cross_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Convert the list of feature columns into a layer that will ultimately become
# part of the model. Understanding layers is not important right now.
buckets_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# The following variables are the hyperparameters.
learning_rate = 0.04
epochs = 35
batch_size = 100
label_name = 'median_house_value'

# Build the model, this time passing in the buckets_feature_layer.
my_model = predefined.create_model(learning_rate, buckets_feature_layer)

# Train the model on the training set.
epochs, rmse = predefined.train_model(my_model, train_df, epochs, batch_size, label_name)

predefined.plot_the_loss_curve(epochs, rmse)

test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name))

print("\n: Evaluate the new model against the test set:")
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)