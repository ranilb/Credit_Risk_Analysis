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

Next the resampled data was trained in logistic regression model and tested for the test data set. The accuracy of this method is 64.56% which implies approximatetly 65 predictions out of 100 is accurate. The result are shown below:

  
  <img width="441" alt="naive_accuracy" src="https://user-images.githubusercontent.com/112113327/213521467-e2287cdd-4c77-44b3-b32b-6b399a4e6641.png">

It can be observed that 53 high risk and 11,675 low risk values are predicted accurately. The table is shown below:

<img width="354" alt="Naive_condussion" src="https://user-images.githubusercontent.com/112113327/213526631-bda4e133-5adc-4f48-a3c1-a500beedcca8.png">

As the last step, the classification report was obtained and shown below:

  <img width="721" alt="naive_classification" src="https://user-images.githubusercontent.com/112113327/213521808-9bde6bb8-d3af-406f-bc47-fd61a9fdddb6.png">
  
  
### SMOTE Oversampling  
To avoid the inbalance situation in the dataset, next we employ the "SMOTE Oversampling" technique. After oversamling the training data set, both low risk and high risk credit records are equal to 51,352. The code and the results are shown below:

  <img width="759" alt="SMOTE_oversampling" src="https://user-images.githubusercontent.com/112113327/213523284-63d2f038-38cc-4c58-a06d-31d0e38be39f.png">

Next the resampled data was trained in logistic regression model and tested on the test dataset. The accuracy of this method is 62.24% which implies approximatetly 62 predictions out of 100 is accurate. The result are shown below:

  
 <img width="356" alt="SMOKE_accuracy" src="https://user-images.githubusercontent.com/112113327/213526032-7fbf87a2-8a84-4dff-a921-83b7c75b6705.png">

It can be observed that 53 high risk and 10,916 low risk values are predicted accurately. The table is shown below:

  <img width="357" alt="SMOKE_confusion" src="https://user-images.githubusercontent.com/112113327/213526504-f77da29a-8a3f-41fb-bfbd-72658046f85e.png">

As the last step, the classification report was obtained and shown below:

 <img width="711" alt="SMOKE_classification" src="https://user-images.githubusercontent.com/112113327/213526553-8b2efb28-c391-4eb3-b955-999639bc3175.png">

