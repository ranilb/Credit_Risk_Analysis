# Credit Risk Analysis

In this work, we analyze the "credit card credit dataset" from LendingClub, a peer-to-peer lending services company. In general, this is an unbalanced classification problem as good loans easily outnumber risky loans. In other words, two high-risk and low-risk credit classes in the dataset aren't equally represented. Therefore, we first employ four diffent resampling techniques: 
* oversample the data using the RandomOverSampler
* oversample the data using the SMOTE
* undersample the data using the ClusterCentroids
* Combination (Over and Under) Sampling
 
 and train them with the logistic model to analyze the credit risk. Then we compare two new machine learning models that reduce bias, "BalancedRandomForestClassifier" and "EasyEnsembleClassifier", to predict credit risk.

#### Data: [Credit card credit dataset](https://github.com/ranilb/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv.zip) from LendingClub



## Results
First, we oversample the data using the "RandomOverSampler" and "SMOTE" algorithms. Then we undersample the data using the "ClusterCentroids" algorithm. finally, we use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. 


### Use Resampling Models to Predict Credit Risk
After cleaning up the dataset, there are 68,470 records for low risk loans and there are only 347 records for high risk loans. Hence it verifies that this is an unbalanced classification problem. There information is shown below:

  <img width="361" alt="balance_target" src="https://user-images.githubusercontent.com/112113327/213509567-d0c94e7e-9e99-4168-83b9-265efd79ad21.png">
  
  
#### <ins> Naive Random Oversampling </ins>
Due to the in balance in the dataset, first employ the "Naive Random Oversampling" technique to balance the low risk and high risk credit records. After oversamling the training data set, both low risk and high risk credit records are equal to 51,352. The code and the results are shown below:

  <img width="536" alt="naive_oversampling" src="https://user-images.githubusercontent.com/112113327/213516236-adb45624-204e-4644-b2ee-538a93ec649e.png">

Next the resampled data was trained in logistic regression model and tested for the test data set. The accuracy of this method is 64.56% which implies approximately 65 predictions out of 100 is accurate. The result are shown below:

  <img width="441" alt="naive_accuracy" src="https://user-images.githubusercontent.com/112113327/213521467-e2287cdd-4c77-44b3-b32b-6b399a4e6641.png">


It can be observed that 53 high risk and 11,675 low risk values are predicted accurately. The table is shown below:

<img width="354" alt="Naive_condussion" src="https://user-images.githubusercontent.com/112113327/213526631-bda4e133-5adc-4f48-a3c1-a500beedcca8.png">


As the last step, the classification report was obtained and shown below:

  <img width="721" alt="naive_classification" src="https://user-images.githubusercontent.com/112113327/213521808-9bde6bb8-d3af-406f-bc47-fd61a9fdddb6.png">
THe precession is 0.01 and sensitivity is 0.61 for this oversampling technique. The precession and sensitivity values do not provide a justice to make good predictions.
  
  
  
#### <ins> SMOTE Oversampling </ins>  
To avoid the inbalance situation in the dataset, next we employ the "SMOTE Oversampling" technique. After oversamling the training data set, both low risk and high risk credit records are equal to 51,352. The code and the results are shown below:

  <img width="759" alt="SMOTE_oversampling" src="https://user-images.githubusercontent.com/112113327/213523284-63d2f038-38cc-4c58-a06d-31d0e38be39f.png">

Next the resampled data was trained in logistic regression model and tested on the test dataset. The accuracy of this method is 62.24% which implies approximatetly 62 predictions out of 100 is accurate. The result are shown below:

 <img width="356" alt="SMOKE_accuracy" src="https://user-images.githubusercontent.com/112113327/213526032-7fbf87a2-8a84-4dff-a921-83b7c75b6705.png">


It can be observed that 53 high risk and 10,916 low risk values are predicted accurately. The table is shown below:

  <img width="357" alt="SMOKE_confusion" src="https://user-images.githubusercontent.com/112113327/213526504-f77da29a-8a3f-41fb-bfbd-72658046f85e.png">


As the last step, the classification report was obtained and shown below:

 <img width="711" alt="SMOKE_classification" src="https://user-images.githubusercontent.com/112113327/213526553-8b2efb28-c391-4eb3-b955-999639bc3175.png">
THe precession is 0.01 and sensitivity is 0.61 for this oversampling technique. The precession and sensitivity and sensitivity values do not provide a justice to make good predictions.



#### <ins> Cluster Centroid Undersampling </ins> 
This techneque is different methos compared to the previous two oversampling methods. In this methos, number of sample data is 260 rows compared to 51,352 in last two methods. Therefore, it is called an undersampling mehod and code is shown below: 

<img width="693" alt="Cluster_count" src="https://user-images.githubusercontent.com/112113327/213530732-9c8400dd-12bc-4a19-a43e-d2e4a022ee2d.png">


Next the resampled data was trained in logistic regression model and tested on the test dataset. The accuracy of this method is 51.28% which implies approximatetly 51 predictions out of 100 is accurate. The result are shown below:

<img width="363" alt="Cluster_accuracy" src="https://user-images.githubusercontent.com/112113327/213530935-0e702269-08d0-4fc9-80b2-6b79e95a9a2b.png">


It can be observed that 50 high risk and 7,717 low risk values are predicted accurately. The table is shown below:

<img width="351" alt="Cluster_confusion" src="https://user-images.githubusercontent.com/112113327/213531848-e1bcd434-89f8-4b16-9c27-2e20f19acf8c.png">


As the last step, the classification report was obtained and shown below:

<img width="713" alt="Cluster_classification" src="https://user-images.githubusercontent.com/112113327/213531911-2ef65f48-00b9-43cf-b27f-6ee3b99efaf6.png">

THe precession is 0.01 and sensitivity is 0.57 for this oversampling technique. The precession and sensitivity values do not provide a justice to make good predictions.



#### <ins> Combination Sampling </ins> 
In the three prevous sections, we discussed about both oversampling and under sampling techneques. Now in this section, we are going to combine both techniques and make a new combination to resample the data. Therefore, in this method, number of shigh-risk data rows is 68,458 and the number of low-risk data rows is 62,022 which are not equal. The code is shown below: 

<img width="692" alt="Combine_count" src="https://user-images.githubusercontent.com/112113327/213536583-35985caa-00c6-4879-b91c-697c950fb345.png">


Next the resampled data was trained in logistic regression model and tested on the test dataset. The accuracy of this method is 65.31% which implies approximatetly 65 predictions out of 100 is accurate. The result are shown below:

<img width="353" alt="combine_accuracy" src="https://user-images.githubusercontent.com/112113327/213536730-d746eefc-f3ac-4319-a250-7a64512408ce.png">


It can be observed that 60 high risk and 10,555 low risk values are predicted accurately. The table is shown below:

<img width="351" alt="combine_confusion" src="https://user-images.githubusercontent.com/112113327/213536925-9aa67a17-f818-41d5-a495-c3df29adb813.png">


As the last step, the classification report was obtained and shown below:

<img width="708" alt="combine_classification" src="https://user-images.githubusercontent.com/112113327/213536964-9353cad8-4827-4f55-a492-70d452393193.png">
THe precession is 0.01 and sensitivity is 0.69 for this oversampling technique. The precession and sensitivity values do not provide a justice to make good predictions.



### Use Ensemble Classifiers to Predict Credit Risk
We now compare two new machine learning models that reduce bias, "BalancedRandomForestClassifier" and "EasyEnsembleClassifier", to predict credit risk, and evaluate the performance of these models.


#### <ins> Balanced Random Forest Classifier </ins> 
In this section, we use a new techneque, called Balanced Random Forest Classifier, to check the accuracy of the predictions.  In this method, number of high-risk data is 260 rows compared to 51,352 of low-risk data. There is no special sampling techneque is used here. However, the data was trained using the  the "BalancedRandomForestClassifier" model rather than the "logistic model".  The accuracy of this method is 78.78% which implies approximatetly 79 predictions out of 100 is accurate. The result are shown below:

<img width="446" alt="brfc_accuracy" src="https://user-images.githubusercontent.com/112113327/213545463-aa74b50f-6436-4c4a-9b49-5683bdb3e2b3.png">


It can be observed that 58 high risk and 15,558 low risk values are predicted accurately. The table is shown below:

<img width="356" alt="brfc_confusion" src="https://user-images.githubusercontent.com/112113327/213545503-5c7034ce-e2d3-430a-9e48-84c42acd97f3.png">


As the last step, the classification report was obtained and shown below:

<img width="712" alt="brfc_classification" src="https://user-images.githubusercontent.com/112113327/213545527-35dd208c-4590-4f44-b1dd-b74d0d9197a3.png">

THe precession is 0.04 and sensitivity is 0.67 for this oversampling technique. The precession and sensitivity values do not provide a justice to make good predictions.


#### <ins> Easy Ensemble AdaBoost Classifier </ins> 
In this last section, we use another techneque, called Easy Ensemble AdaBoost Classifie, to check the accuracy of the predictions.  In this method, number of high-risk data is 260 rows compared to 51,352 of low-risk data. There is no special sampling techneque is used here. However, 
the data was trained using the "EasyEnsembleClassifier" model.  The accuracy of this method is 92.54% which implies approximatetly 93 predictions out of 100 is accurate and which is great. The result are shown below:

<img width="361" alt="easy_accuracy" src="https://user-images.githubusercontent.com/112113327/213549201-6ed56fa2-63a7-4db4-899d-d0da144000b2.png">


It can be observed that 79 high risk and 16,139 low risk values are predicted accurately. The table is shown below:

<img width="355" alt="easy_confusion" src="https://user-images.githubusercontent.com/112113327/213549327-05271ff1-2076-428a-84c1-4946872fbb15.png">


As the last step, the classification report was obtained and shown below:

<img width="716" alt="easy_classification" src="https://user-images.githubusercontent.com/112113327/213549383-fdb7f689-111a-41df-a39e-72fd35b7bfef.png">
THe precession is 0.07 and sensitivity is 0.91 for this oversampling technique. The precession and sensitivity values have improved and provide a justice to make good predictions.


## Summary
Three different prediction models were performed on the dataset to determine if a credit risk is high. The Ensemble models brought a lot more improvment specially on the sensitivity of the high risk credits. The data was trained using logistic model with three different sampling technique, however, the accuracy, precession and the sensitivity values are not in a higher level.  Then the "Balanced Random Forest Classifier" model showed some improvements with respect to the accuracy, precession and the sensitivity values, but the "Easy Ensemble AdaBoost Classifier" model outperformed in all aspects.

The accuracy ot the Easy Ensemble AdaBoost Classifier is about 93% which is not perfect, but great among the tested models. Therefore, I would recommend the bank to use this model to predict credit risk, if they can improve the accuracy by resampling the training data using the methods discussed in the previous section.
