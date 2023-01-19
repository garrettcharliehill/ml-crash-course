import pandas as pd
import predefined

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Import the dataset.
training_df = pd.read_csv(filepath_or_buffer="./datasets/california_housing_train.csv")

# Scale the label.
training_df["median_house_value"] /= 1000.0

# Print the first rows of the pandas DataFrame.
print(training_df.head())

# Get statistics on the dataset.
print(training_df.describe())

training_df["rooms_per_person"] = training_df['total_rooms'] / training_df['population']

# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 30
batch_size = 30

# Specify the feature and the label.
my_feature = "rooms_per_person"
my_label="median_house_value" # the median value of a house on a specific city block.

# Discard any pre-existing version of the model.
my_model = None

# Invoke the functions.
# my_model = predefined.build_model(learning_rate)
# weight, bias, epochs, rmse = predefined.train_model(
#     my_model,
#     training_df, 
#     my_feature,
#     my_label,
#     epochs,
#     batch_size
# )

# print("\nThe learned weight for your model is %.4f" % weight)
# print("The learned bias for your model is %.4f\n" % bias )

# predefined.plot_the_model(training_df, weight, bias, my_feature, my_label)
# predefined.plot_the_loss_curve(epochs, rmse)

# predefined.predict_house_values(training_df, my_model, 10, my_feature, my_label)

print(training_df.corr())