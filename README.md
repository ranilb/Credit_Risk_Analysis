# Credit Risk Analysis

In this work, we analyze the "credit card credit dataset" from LendingClub, a peer-to-peer lending services company. In general, this is an unbalanced classification problem as good loans easily outnumber risky loans. In other words, two good and bad loan classes in the dataset aren't equally represented. Therefore, we employ three different techniques in machine learning to train and evaluate models with unbalanced classes to determine which is better at predicting credit risk.

First, we oversample the data using the "RandomOverSampler" and "SMOTE" algorithms. Then we undersample the data using the "ClusterCentroids" algorithm. finally, we use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Once we create training and test data from the above methods, we compare two new machine learning models that reduce bias, "BalancedRandomForestClassifier" and "EasyEnsembleClassifier", to predict credit risk, and evaluate the performance of these models.


#### Data: [Credit card credit dataset](https://github.com/ranilb/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv.zip) from LendingClub

## Results
After cleaning up the dataset, there are 68,470 records for low risk loans and there are only 347 records for high risk loans. Hence it veryfies that this is an unbalanced classification problem. There information is shown below:

  <img width="361" alt="balance_target" src="https://user-images.githubusercontent.com/112113327/213509567-d0c94e7e-9e99-4168-83b9-265efd79ad21.png">
  
### Naive Random Oversampling
Due to the inbalance in the dataset, first employ the "Naive Random Oversampling" technique to balance the low risk and high risk credit records. After oversamling the training data set, both low risk and high risk credit records are equal to 51,352. The code and the results are shown below:

  <img width="536" alt="naive_oversampling" src="https://user-images.githubusercontent.com/112113327/213516236-adb45624-204e-4644-b2ee-538a93ec649e.png">


