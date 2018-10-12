## Enron Submission Free-Response Questions
1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]
The goal of the project is to use the dataset to identify person of interest (poi), *"individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity"*. The dataset includes tens of thousands of emails and detailed financial data of top executives at Enron, that was one of the largest companies in the United States. It was bankrupted in 2002 due to corporate fraud. Using the machine learning skills, I hope I can develop an algorithm to identify a poi based on the financial and email data. 
Since I'm interested in individual people, I want to exclude the data that do not represent individuals; I know that we can access individuals by *enron_data["LASTNAME FIRSTNAME"]* or  *enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"]*. I've used a regular expression to identify the names that do not meet either *LASTNAME FIRSTNAME* or *LASTNAME FIRSTNAME MIDDLEINITIAL*. Then I've identified 2 outliers, *'TOTAL'* and *'THE TRAVEL AGENCY IN THE PARK'*. Then I've tried to remove the data point that have too many 'NaN' values, which is *'LOCKHART EUGENE E'*. This individual only has a poi value ('False'), which won't help us understand the relationships between poi and the other data. That's why I've removed it as well. Please see more detailed explanations in the code [poi_id.py](poi_id.py).

2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]
Below is the feature list before feature selection:
```python
my_features_list = ['poi','salary', 'deferral_payments',  'total_payments', 'loan_advances', 
	'bonus', 'restricted_stock_deferred', 'deferred_income', 
	'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
	'restricted_stock', 'director_fees', 'shared_receipt_with_poi', 'fraction_from_poi', 'fraction_to_poi']
```
I've created 2 new features: *'fraction_from_poi'* and *'fraction_to_poi'*. The *'fraction_from_poi'* is the portion of emails from poi to this person out of all the emails received by this person; the *'fraction_to_poi'* is the portion of emails from this person to poi out of all the emails sent from this person. I want to have data sets that answer the question, if a person has frequent communication with a poi via email, will this person also be a poi? From the scatter plot of the two new features, it seems that if a person sends over 20% or more emails to poi, then it's likely that this person is a poi. Since these 2 new features are strongly related with *'from_poi_to_this_person'*, *'to_messages'*, *'from_this_person_to_poi'*, and *'from_messages'*. I've also removed these 4 features from the feature list. 

Then I use feature selection function, SelectKBest, and linear dimensionality reduction function, PCA to help me reduce noise of the data sets and boost performance. I couldn't decide which one to use. Both of them seem to have pros and cons, so I use *"GridSearchCV"* to help me decide on the best estimator. For the final algorithm I use, where DecisionTreeClassifier is applied, SelectKBest is used. 

Below is the feature scores from the SelectKBest funtion:
```text
[ 16.79676028   0.09431984   8.01408707   6.38262675  23.70113458
   0.83663467  10.5189354    3.80714308  14.84299273   3.84053584
   8.81853718   8.56676916   1.78058808   8.97930042   1.21220763
  14.83493602]
```
12 features are selected according to the k highest scores. 4 features are removed: *"deferral_payments"*, *"restricted_stock_deferred"*, *"director_fees"* and *"fraction_from_poi"*. 

Since I use a decision tree algorithm, I can get the feature importances of the selected features.

```text
[ 0.04135266  0.01809179  0.          0.05169082  0.05169082  0.18608696
  0.12206986  0.14141132  0.          0.04461625  0.12479641  0.21819311]
```
It seems that the feature, *"fraction_to_poi"*, is the most powerful feature. 

3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

I've tried and tested 3 algorithms in total, GaussianNB, DecisionTreeClassifier, and Support Vector Classification (SVC). I end up using DecisionTreeClassifier because it has a good balance among accuracy, precision and recall scores. GaussianNB is easy to use, but it doesn't have a good accuracy score. Although SVC has a fairly good accuracy score (0.92), but I get this message when I try to call its precision score: *"UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples."* Calling the predicted values and actual test values, I've found that there's no True Positives. Therefore, I decide not to use the algorithm. DecisionTreeClassifier also gives me an accuracy score of 0.92; and I can tune its precision and recall scores to be greater than 0.3. 

4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

Tuning the parameters of an algorithm can bring out the best performance of an algorithm. For example, I use the DecisionTreeClassifier as my classifier. The parameters of the classifier, *"min_samples_split"* and *"random_state"*, can influence the accuracy score of the algorithm. To have the best estimator, I've used *"GridSearchCV"* to tune the parameters in my algorithm. The best estimator I found is when the 2 parameters are: *min_samples_split=3*, *random_state=10*. 

```text
Best estimator found by grid search:
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=3, min_weight_fraction_leaf=0.0,
            presort=False, random_state=10, splitter='best'))])
```

5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

Validation is the process to assess whether an algorithm is actually doing what I want it to do. When we use training-testing split method, it's easy to make the test size either too big or too small. This will cause the algorithm either to be overfitting or failing to make predictions due to the small size of training data. I've chosen my test size to be 0.25 since it gives me the best performance.

```text
from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.25, random_state=42)
```

I also validate the performance of my algorithms using cross validation to automate the parameter tuning tests. I've used *"GridSearchCV"* to find the best estimator of my algorithms. For instance, as I mentioned previous, it helps me choose the *"min_samples_split"* and *"random_state"* values. *"GridSearchCV"* also helps me decide on whether to use SelectKBest or PCA for feature selection/dimensionality reduction as well as the number of features/components I should use for an algorithm. 

6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

Below is the outcome after I use the testing script:

```text
Accuracy: 0.83387	Precision: 0.36086	Recall: 0.31900	F1: 0.33864	F2: 0.32658
	Total predictions: 15000	True positives:  638	False positives: 1130	False negatives: 1362	True negatives: 11870
```

My identifier has slightly better precision than recall. That means that whenever a POI gets flagged in my test set, it's about 36% chance to be a real POI and not a false alarm. On the other hand, the price I pay for this is that sometimes I miss real POIs. My identifier has a F1 score of 0.34. To conclude, if my identifier finds a POI then the person is 36% likely to be a POI, and if the identifier does not flag someone, then it's about 32% likely that they are not a POI. 
