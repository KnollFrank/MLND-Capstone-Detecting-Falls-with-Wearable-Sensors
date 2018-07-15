# adapted from projects finding_donors and customer_segments
###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def distribution(data, transformed=False):
    """
    Visualization code for displaying skewed distributions of features
    """

    # Create figure
    fig = pl.figure(figsize=(11, 5));

    # Skewed feature plotting
    for i, feature in enumerate(['capital-gain', 'capital-loss']):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.hist(data[feature], bins=25, color='#00A0A0')
        ax.set_title("'%s' Feature Distribution" % (feature), fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
                     fontsize=16, y=1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
                     fontsize=16, y=1.03)

    fig.tight_layout()
    fig.show()


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # Create figure
    fig, ax = pl.subplots(2, 3, figsize=(11, 7))

    # Constants
    bar_width = 0.25
    colors = ['#A00000', '#00A0A0', '#00A000', 'yellow']

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                # Creative plot code
                ax[j // 3, j % 3].bar(i + k * bar_width, results[learner][i][metric], width=bar_width, color=colors[k])
                ax[j // 3, j % 3].set_xticks([0.45, 1.45, 2.45])
                ax[j // 3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j // 3, j % 3].set_xlabel("Training Set Size")
                ax[j // 3, j % 3].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[0, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner))
    pl.legend(handles=patches, bbox_to_anchor=(-.80, 2.53), \
              loc='upper center', borderaxespad=0., ncol=4, fontsize='x-large')

    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize=16, y=1.10)
    # pl.tight_layout()
    pl.show()


def feature_plot(importances, X_train, num_features):
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:num_features]]
    values = importances[indices][:num_features]

    # Creat the plot
    fig = pl.figure(figsize=(9, 5))
    pl.title("Normalized Weights for First " + str(num_features) + " Most Predictive Features", fontsize=16)
    pl.bar(np.arange(num_features), values, width=0.6, align="center", color='#00A000', \
           label="Feature Weight")
    pl.bar(np.arange(num_features) - 0.3, np.cumsum(values), width=0.2, align="center", color='#00A0A0', \
           label="Cumulative Feature Weight")
    pl.xticks(np.arange(num_features), columns)
    pl.xlim((-0.5, num_features - .5))
    pl.ylabel("Weight", fontsize=12)
    pl.xlabel("Feature", fontsize=12)

    pl.legend(loc='upper center')
    pl.tight_layout()
    pl.show()
    return columns


def pca_results(good_data, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1, len(pca.components_) + 1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns=list(good_data.keys()))
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the feature weights as a function of the components
    components.plot(ax=ax, kind='bar');
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i - 0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f" % (ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis=1)


from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from time import time


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test, beta):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    start = time()
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()

    results['train_time'] = end - start

    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()

    results['pred_time'] = end - start

    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    results['acc_test'] = accuracy_score(y_test, predictions_test)

    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=beta)

    results['f_test'] = fbeta_score(y_test, predictions_test, beta=beta)

    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    return results
