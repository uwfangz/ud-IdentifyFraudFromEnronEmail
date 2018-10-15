#!/usr/bin/python
import matplotlib.pyplot
import sys
import pickle
import numpy as np
import re
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Initial selection of features
#==============================================================================
features_list = ['poi','salary', 'deferral_payments',  'total_payments', 'loan_advances', 
	'bonus', 'restricted_stock_deferred', 'deferred_income', 
	'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
	'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features
# There are 19 features. I didn't include 'email_address' \
# since it's unique value.
#==============================================================================

### Revised features_list
my_features_list = ['poi','salary', 'deferral_payments',  'total_payments', 'loan_advances', 
	'bonus', 'restricted_stock_deferred', 'deferred_income', 
	'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
	'restricted_stock', 'director_fees', 'shared_receipt_with_poi', 'fraction_from_poi', 'fraction_to_poi'] # Added 2 new features
	
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 1-1: Data Exploration
#==============================================================================
## total number of data points: 146
print "Total Number Of Data Points:", len(data_dict)
#==============================================================================
## allocation across classes (POI/non-POI): 18/128
poi = 0
non_poi = 0
for name in data_dict:
	if data_dict[name]['poi'] == False:
		non_poi += 1
	elif data_dict[name]['poi'] == True:
		poi += 1
print "Allocation Across Classes (poi/non-poi):", poi, non_poi
# print poi, non_poi
#==============================================================================
## number of features used (use 'GLISAN JR BEN F' as an example data point): 21
print "Number Of Features Used:", len(data_dict['LAY KENNETH L'])
#==============================================================================
## are there features with many missing values? etc.: ('loan_advances', 142)
nan_dict = {}
for name in data_dict:
	for f in data_dict[name]:
		if f not in nan_dict and data_dict[name][f] == 'NaN':
			nan_dict[f] = 1
		elif f in nan_dict and data_dict[name][f] == 'NaN':
			nan_dict[f] += 1
# print nan_dict			
# 
## find the max value in nan_dict

maximum = max(nan_dict, key=nan_dict.get) 
# print(maximum, nan_dict[maximum])
print "Are there features with many missing values?", (maximum, nan_dict[maximum])

## The feature, 'loan_advances', has the most missing values. \
## In fact, only 4 data points do not have missing values of 'loan_advances'.

print "Data points do not have missing values of 'loan_advances':"
for name in data_dict:
	if data_dict[name]['loan_advances'] != 'NaN':
		print (name, data_dict[name]['poi'], data_dict[name]['loan_advances'])
		
#==============================================================================

### Task 2: Remove outliers

### Since we are interested in individual people, \
### we want to exclude the data that do not represent \
### individuals; we know we can access individual people \
### by enron_data["LASTNAME FIRSTNAME"] or  enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"]

### In this section, I'll use a regular expression to exclude \
### name in data_dict that do not meet "LASTNAME FIRSTNAME" or "LASTNAME FIRSTNAME MIDDLEINITIAL"

### Use regex to find name outliers
#print "Names that do not match 'LASTNAME FIRSTNAME' or 'LASTNAME FIRSTNAME MIDDLEINITIAL' : "
#==============================================================================
# for name in data_dict:
# 	m = re.match(r"^([A-Z]+(-[A-Z]+)?)\s([A-Z]+)(\s[A-Z])?$", name)
# 	if not m:
# 		print name
#==============================================================================		
### NOT matched names
#==============================================================================
## WALLS JR ROBERT H
## BOWEN JR RAYMOND M
## OVERDYKE JR JERE C
## PEREIRA PAULO V. FERRAZ
## BLAKE JR. NORMAN P
## THE TRAVEL AGENCY IN THE PARK
## TOTAL
## WHITE JR THOMAS E
## WINOKUR JR. HERBERT S
## GARLAND C KEVIN
## YEAGER F SCOTT
## DERRICK JR. JAMES V
## DONAHUE JR JEFFREY M
## GLISAN JR BEN F
#==============================================================================

### There are still a lot of individual names in the result. However, \
### we can clearly tell what are the outliers of names: "TOTAL", \
### and "THE TRAVEL AGENCY IN THE PARTK".

### From the Enron Outlier project, we know that 'TOTAL' is an outlier
### Let's take a look at 'THE TRAVEL AGENCY IN THE PARK'
#==============================================================================
# print data_dict['THE TRAVEL AGENCY IN THE PARK']
#==============================================================================
### A lot of feature values are displayed as 'NaN' and it's not 'poi'
### Similarly, we should remove the data that have many 'NaN' values.

### There are 21 features in the whole dataset.
### To be conservative, let's check and see \
### what individual people have >= 20 features that are 'NaN'
#==============================================================================
for name in data_dict:
	n_value = 0
	for f in data_dict[name]:
		if data_dict[name][f] == 'NaN':
			n_value += 1
	if n_value >= 20:
		print "Individual(s) have greater than 20 'NaN' features: ", name
#==============================================================================
### We get 'LOCKHART EUGENE E'

# print data_dict['LOCKHART EUGENE E']

### We only know that 'LOCKHART EUGENE E' is not a poi; \
### we don't have other information on the person.

### Pop out the outliers
#==============================================================================
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
#==============================================================================

### Task 3: Create new feature(s)
### Create fraction of 'from_poi_to_this_person' / 'to_messages'
### Create fraction of 'from_this_person_to_poi' / 'from_messages'

### Set a function to compute fraction:

def computeFraction( spec, total ):
    fraction = 0.
    if spec == "NaN" or total == "NaN":
        fraction = 0.
    else:
        fraction = float(spec) / total

    return fraction

### Add new features to data_dict
new_features = {}
for name in data_dict:

    data_point = data_dict[name]
	
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi
    

### Add new features to feature_list
 
### Store to my_dataset for easy export below.

my_dataset = data_dict

### Extract features and labels from dataset for local testing
### Update features_list to my_features_list

### Compare initial features_list with my_features_list (including 2 new features)
#==============================================================================
# data = featureFormat(my_dataset, features_list, sort_keys = True)

data = featureFormat(my_dataset, my_features_list, sort_keys = True)
#==============================================================================

labels, features = targetFeatureSplit(data)

### Feature selection using SelectKBest
#==============================================================================
# from sklearn.feature_selection import SelectKBest, f_classif
# selector = SelectKBest(f_classif, k=10)
# selector.fit_transform(features, labels)
#==============================================================================

### Scale features
#==============================================================================
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit_transform(features)
#==============================================================================

### Visualization two new features
#==============================================================================
# for point in data:
#     fraction_from_poi = point[-2]
#     fraction_to_poi = point[-1]
#     if point[0] == 0:
#     	matplotlib.pyplot.scatter(fraction_from_poi, fraction_to_poi, color = "b")
#     else:
#     	matplotlib.pyplot.scatter(fraction_from_poi, fraction_to_poi, color = "r")
#     	
# matplotlib.pyplot.xlabel("fraction_from_poi")
# matplotlib.pyplot.ylabel("fraction_to_poi")
# matplotlib.pyplot.show()
#==============================================================================

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### training-testing split needed in regression, just like classification
from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.25, random_state=42)

### Use PCA and SelectKBest for feature selection/dimensionality reduction ###
#==============================================================================
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#==============================================================================

### GaussianNB Clf ###
#==============================================================================
# pipe = Pipeline([('reduce_dim', PCA()), ('clf', GaussianNB())])
# N_FEATURES_OPTIONS = [2, 4, 5, 8]
# 
# param_grid = [
#     {
#     	'reduce_dim': [PCA()],
#         'reduce_dim__n_components': N_FEATURES_OPTIONS,
#     },
#     {
#         'reduce_dim': [SelectKBest(f_classif)],
#         'reduce_dim__k': N_FEATURES_OPTIONS,
#     },
# ]
# 
# clf = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid)
# 
# clf.fit(feature_train, label_train)
# pred = clf.predict(feature_test)
# 
# print "Best estimator found by grid search:"
# print clf.best_estimator_
# 
# accuracy = accuracy_score(label_test, pred)
# print accuracy

## accuracy = 0.89

#==============================================================================

### DecisionTreeClassifier ###
#==============================================================================
pipe = Pipeline([('reduce_dim', PCA()), ('clf', DecisionTreeClassifier())])
N_FEATURES_OPTIONS = [4, 6, 8, 10, 12]
MIN_SAMPLES_SPLIT_OPTIONS = [2,3,4,5]
RANDOM_STATE_OPTIONS = [5,10,15,20,40]

param_grid = [
     {
     	'reduce_dim': [PCA()],
         'reduce_dim__n_components': N_FEATURES_OPTIONS,
         'clf__min_samples_split': MIN_SAMPLES_SPLIT_OPTIONS,
         'clf__random_state': RANDOM_STATE_OPTIONS
     },
     {
         'reduce_dim': [SelectKBest(f_classif)],
         'reduce_dim__k': N_FEATURES_OPTIONS,
         'clf__min_samples_split': MIN_SAMPLES_SPLIT_OPTIONS,
         'clf__random_state': RANDOM_STATE_OPTIONS
     },
 ]

clf = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid)

clf.fit(feature_train, label_train)

pred = clf.predict(feature_test)

# print "Best estimator found by grid search:"
clf = clf.best_estimator_

# print clf.named_steps['reduce_dim'].get_support()
## [ True False  True  True  True False  True  True  True  True  True  True
## False  True False  True]

print "Feature scores using SelectKBest function: "
for i in range(len(my_features_list)-1):
	print my_features_list[i+1] + ": ", clf.named_steps['reduce_dim'].scores_[i]

print "Features selected using SelectKBest function: "
for i in range(len(my_features_list)-1):
	if clf.named_steps['reduce_dim'].get_support()[i]:
		print my_features_list[i+1]

accuracy = accuracy_score(label_test, pred)
print "Accuracy score of DecisionTreeClassifier: ", accuracy

## accuracy = 0.92
#==============================================================================

### DecisionTree feature selection (feature_importances_)###
#==============================================================================
important_features = clf.named_steps['clf'].feature_importances_
print "Feature importances of the selected features: "
print important_features
# for n in range(len(important_features)): # 16 important features
# 	if important_features[n] > 0.1:
# 		print n
		# print important_features[n]

#==============================================================================

### SVC ###
#==============================================================================
# pipe = Pipeline([('reduce_dim', PCA()), ('clf', SVC(kernel='rbf'))])
# N_FEATURES_OPTIONS = [2, 4, 8]
# C_OPTIONS = [0.001, 0.01, 0.1, 1, 10, 100]
# GAMMA_OPTIONS = [0.001, 0.01, 0.1, 1]
# param_grid = [
#     {
#     	'reduce_dim': [PCA()],
#         'reduce_dim__n_components': N_FEATURES_OPTIONS,
#         'clf__C': C_OPTIONS,
#         'clf__gamma': GAMMA_OPTIONS
#     },
#     {
#         'reduce_dim': [SelectKBest(f_classif)],
#         'reduce_dim__k': N_FEATURES_OPTIONS,
#         'clf__C': C_OPTIONS,
#         'clf__gamma': GAMMA_OPTIONS
#     },
# ]
# 
# clf = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid)
# 
# clf.fit(feature_train, label_train)
# 
# pred = clf.predict(feature_test)
# 
# print "Best estimator found by grid search:"
# clf = clf.best_estimator_
# 
# print pred
# 
# accuracy = accuracy_score(label_test, pred)
# print accuracy
# 
# from sklearn.metrics import precision_score
# print precision_score(label_test, pred)

#==============================================================================
### Use precision_score to evaluate performance
# from sklearn.metrics import precision_score
# print precision_score(label_test, pred)
#==============================================================================



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_features_list)