import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import networkx as nx
import os
import re
import csv
from datetime import datetime

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.ensemble import BalancedBaggingClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate, cross_val_predict, KFold
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample

import umap

#################################################################################
# Plot Data in 2 Dimentions
#################################################################################


def plot_pca(X, y):
    # Convert y to numeric if it contains non-numeric values
    if not all(type(val) in [int, float] for val in y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in set(y):
        indices = y == label
        ax.scatter(X_pca[indices, 0], X_pca[indices, 1], label=label, alpha=0.5)
    
    ax.legend()
    plt.show()


def plot_umap(X, y):
    
    import umap
    
    # Convert y to numeric if it contains non-numeric values
    if not all(type(val) in [int, float] for val in y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Apply UMAP
    umap_model = umap.UMAP(n_components=2)
    X_umap = umap_model.fit_transform(X)

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in set(y):
        indices = y == label
        ax.scatter(X_umap[indices, 0], X_umap[indices, 1], label=label, alpha=0.5)
    
    ax.legend()
    plt.show()


#################################################################################
# Plot Comfusion Matrix
#################################################################################

def plot_cfm(cfm,
             normalize=False,
             cmap=plt.cm.Blues):

    if normalize:
        cfm = (cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]) * 100

    plt.figure(figsize=(5, 5))
    plt.imshow(cfm, interpolation='nearest', cmap=cmap)

    fmt = '{:.0f}%' if normalize else '{:.0f}'
    thresh = cfm.max() / 2.

    for i in range(cfm.shape[0]):
        for j in range(cfm.shape[1]):
            plt.text(j, i, fmt.format(cfm[i, j]),  # Use '{:.0f}%' for formatting
                     ha='center', va='center',
                     color='white' if cfm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('\nPredicted label')
    plt.tight_layout()
    plt.show()



#################################################################################
# Calculate Micro & Macro Metrics
#################################################################################

def print_classification_metrics(y_true_classes, y_pred_classes):
    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Get the metrics for each class
    precision, recall, f1, support = precision_recall_fscore_support(y_true_classes, y_pred_classes, average=None, zero_division=1)

    # Calculate accuracy per class
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)

    # Print metrics for each class with individual Accuracy
    print('Class Performance Metrics \n')
    print(f"{'Class':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10} | {'Support':<10}")
    for i in range(len(precision)):
        print(f"{i:<12} | {accuracy_per_class[i]:<10.2f} | {precision[i]:<10.2f} | {recall[i]:<10.2f} | {f1[i]:<10.2f} | {support[i]:<10}")

    print("\n----------------------------------------------------------------\n")

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_true_classes, y_pred_classes)

    # Get the metrics for all classes (macro average)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, average='macro')

    # Print macro metrics with overall Accuracy
    print('Model Performance Metrics \n')
    print(f"{'Macro':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10} ")
    print(f"{'':<12} | {overall_accuracy:<10.2f} | {precision_macro:<10.2f} | {recall_macro:<10.2f} | {f1_macro:<10.2f}")



#################################################################################
# Calculate Baseline Loss
#################################################################################


def cv_loss(model, X, y):
    return -round(np.mean(cross_validate(model, X, y,  scoring = 'neg_log_loss', cv=10)['test_score']),3)



def baseline_cross_val(X, y):
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    clf = LogisticRegression(class_weight = 'balanced', solver='lbfgs', penalty='l2', max_iter=1000)    
    clf.fit(X, y)    
    
    print("Clasifier Log Loss:",cv_loss(clf, X, y))
    print(classification_report(y, cross_val_predict(clf, X, y, cv=skf)))



#################################################################################
# Write Submission Results Function
#################################################################################

def write_submission(filename, predictions, test_df):
    # Define the directory for saving submissions
    submission_dir = 'submissions'

    # Create the submissions directory if it doesn't exist
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)

    # Define the file path for saving the submission
    submission_file = os.path.join(submission_dir, filename)

    # Write predictions to the submission file
    with open(submission_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        header = ['domain_name'] + ['class_' + str(i) for i in range(9)]
        writer.writerow(header)
        for i, test_host in enumerate(test_df['domain']):
            row = [test_host] + predictions[i, :].tolist()
            writer.writerow(row)

    print("Submission file saved:", submission_file)









