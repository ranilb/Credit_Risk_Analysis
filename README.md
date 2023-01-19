# Credit Risk Analysis

In this work, we analyze the "credit card credit dataset" from LendingClub, a peer-to-peer lending services company. In general, this is an unbalanced classification problem as good loans easily outnumber risky loans. In other words, two good and bad loan classes in the dataset aren't equally represented. Therefore, we employ three different techniques in machine learning to train and evaluate models with unbalanced classes.

First, we oversample the data using the "RandomOverSampler" and "SMOTE" algorithms. Then we undersample the data using the "ClusterCentroids" algorithm. finally, we use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Once we create training and test data from the above methods, we compare two new machine learning models that reduce bias, "BalancedRandomForestClassifier" and "EasyEnsembleClassifier", to predict credit risk, and evaluate the performance of these models.
