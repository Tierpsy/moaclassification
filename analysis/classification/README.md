# Classification analysis script

The main script for the classification of compounds in modes of action. Produces the results for figure 3(B) of the paper and supplementary figures [--].

It takes as input the smoothed and balanced data in data/smothed_data and performs the following steps:
- Read the data.
- Select feature sets of different sizes using recursive feature elimination and choose the number of features that gives the best cross-validation accuracy using majority vote to get the compound-level predictions.
- Using the selected feature set, optimize the estimator hyperparameters based on cross validation accuracy.
- Get the best estimator trained with the entire training set and predict the mode of action of the compounds in the test set using majority vote to get compound-level predictions.
