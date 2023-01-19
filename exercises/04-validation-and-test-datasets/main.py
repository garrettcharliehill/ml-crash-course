import pandas as pd
import predefined
import numpy as np

train_df = pd.read_csv("./datasets/california_housing_train.csv")
test_df = pd.read_csv("./datasets/california_housing_test.csv")

scale_factor = 1000.0

# Scale the training set's label.
train_df["median_house_value"] /= scale_factor 

# Scale the test set's label
test_df["median_house_value"] /= scale_factor

# The following variables are the hyperparameters.
learning_rate = 0.08
epochs = 30
batch_size = 100

# Split the original training set into a reduced training set and a
# validation set. 
validation_split = 0.2

# Identify the feature and the label.
my_feature = "median_income"    # the median income on a specific city block.
my_label = "median_house_value" # the median house value on a specific city block.
# That is, you're going to create a model that predicts house value based 
# solely on the neighborhood's median income.  

shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))

# Invoke the functions to build and train the model.
my_model = predefined.build_model(learning_rate)
epochs, rmse, history = predefined.train_model(
    my_model,
    shuffled_train_df,
    my_feature, 
    my_label,
    epochs,
    batch_size, 
    validation_split
)

predefined.plot_the_loss_curve(
    epochs,
    history["root_mean_squared_error"], 
    history["val_root_mean_squared_error"]
)

x_test = test_df[my_feature]
y_test = test_df[my_label]

results = my_model.evaluate(x_test, y_test, batch_size=batch_size)
print(results)