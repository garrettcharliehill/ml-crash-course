import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import predefined

my_feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
my_label   = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

# hyperparameters 

# learning_rate=0.01
# epochs=50
# my_batch_size=12

# learning_rate=100 
# epochs=500 
# my_batch_size=12

# learning_rate=0.12 
# epochs=75
# my_batch_size=12

learning_rate=0.05
epochs=100
my_batch_size= 1

# model
my_model = predefined.build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = predefined.train_model(
    my_model,
    my_feature, 
    my_label, epochs,
    my_batch_size
)

# plot results
predefined.plot_the_model(trained_weight, trained_bias, my_feature, my_label)
predefined.plot_the_loss_curve(epochs, rmse)