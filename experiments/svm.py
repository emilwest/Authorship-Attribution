# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:57:39 2020

@author: emwe9516
"""
#import sklearn
from sklearn import svm
from sklearn import metrics
import pandas as pd # for reading file
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import rpy2.robjects as robjects
from timeit import default_timer as timer


robjects.r['load']("C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\dtm_train.RData")
dtm_train = robjects.r['dtm_train']
dtm_train = robjects.r['as.matrix'](dtm_train)
dtm_train = np.array(dtm_train)


robjects.r['load']("C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\dtm_test.RData")
dtm_test = robjects.r['dtm_test']
dtm_test = robjects.r['as.matrix'](dtm_test)
dtm_test = np.array(dtm_test)


#import sys
#print("Python version")
#print (sys.version)

#print('The scikit-learn version is {}.'.format(sklearn.__version__))
# The scikit-learn version is 0.21.3

#data = pd.read_csv('C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\dtm_train.txt', sep=" ")
#data_test = pd.read_csv('C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\dtm_test.txt', sep=" ")

#dtm_train = data.to_numpy()
#dtm_test = data_test.to_numpy()


ytrain = pd.read_csv('C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\y.train', sep=" ")
ytrain = ytrain['x'].astype('category')

ytest = pd.read_csv('C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\y.test', sep=" ")
ytest = ytest['x'].astype('category')



#def my_kernel(X, Y):
#    return np.dot(X, Y.T)


#lin = svm.LinearSVC()
#lin.fit(dtm_train, ytrain)
#preds = lin.predict(dtm_test)
#metrics.accuracy_score(ytest, preds)

##################################################
## Testing

#clf = svm.SVC(decision_function_shape='ovo', kernel=my_kernel)
clf = svm.SVC(decision_function_shape='ovo', kernel="linear", C=200, cache_size=800)
clf.fit(dtm_train, ytrain)
preds = clf.predict(dtm_test)
metrics.accuracy_score(ytest, preds) # .4612
# .6476 for C=100
ytest.cat.categories
print(classification_report(ytest, preds, target_names=ytest.cat.categories))




############################################
# using kernel matrix
start = timer()
gram = np.dot(dtm_train, dtm_train.T)
end = timer()
print(end - start) 

kernel_test = dtm_test@dtm_train.T # dot product

clfprecom = svm.SVC(decision_function_shape='ovo', kernel='precomputed', C=200, cache_size=800)
clfprecom.fit(gram, ytrain)
preds = clfprecom.predict(kernel_test)
preds = pd.Series(preds)
metrics.accuracy_score( ytest , preds) 
# .6476 for C=100



##################################################
# GRID SEARCH CROSS VALIDATION
# https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
# https://stackoverflow.com/questions/24595153/is-it-possible-to-tune-parameters-with-grid-search-for-custom-kernels-in-scikit

# Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [2e-4, 2e-3, 2e-2, 2e-1, 2e0, 2e1, 2e2, 2e3, 2e4]}]


tuned_parameters = [  {'kernel': ['linear'], 'C': [ 2e-1, 2e0, 2e1, 2e2, 2e3, 2e4]} ]

#scores = ['precision', 'recall']
scores = ['precision']
start = timer()
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        svm.SVC(decision_function_shape='ovo', kernel='precomputed'), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(gram, ytrain)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = ytest, clf.predict(kernel_test)
    print(classification_report(y_true, y_pred))
    print()

end = timer()
print(end - start) 

clf.best_params_['C']

clfprecom = svm.SVC(decision_function_shape='ovo', kernel='precomputed', C=clf.best_params_['C'], cache_size=800)

#clfprecom = svm.SVC(decision_function_shape='ovo', kernel='precomputed', C=200, cache_size=800)
#gram = np.dot(dtm_train, dtm_train.T)

clfprecom.fit(gram, ytrain)
#kernel_test = dtm_test@dtm_train.T # dot product
preds = clfprecom.predict(kernel_test)
preds = pd.Series(preds)
metrics.accuracy_score( ytest , preds)  #.6464


confmat = confusion_matrix(ytest, preds)
report = classification_report(ytest, preds, target_names=ytest.cat.categories,  output_dict=True)
print(classification_report(ytest, preds, target_names=ytest.cat.categories) )
report_df =  pd.DataFrame(report).transpose()
bestC = {'bestC' :  [clf.best_params_['C']]}
pd.DataFrame.from_dict(bestC)
print(bestC)

report_df.to_csv('C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\report.csv')
np.savetxt('C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\confmat.csv', np.asarray(confmat), delimiter=",", fmt='%1i')
pd.DataFrame.from_dict(bestC).to_csv('C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\bestC.csv')


clfprecom.support_.size




#dtm_train[clfprecom.support_  , :] == dtm_train
#rounded1 = np.around(dtm_train, 3)
#rounded2 = np.around(dtm_train[clf.support_  , :], 3)
#np.allclose(rounded1, rounded2)
#rounded1.all() == rounded2.all()
#dtm_train[clf.support_, :].all() == dtm_train.all()
#they differ by 0.1
#np.allclose(dtm_train[clfprecom.support_, :] , dtm_train, rtol=1e-01,atol=1e-02)


#b = dtm_train[clfprecom.support_ , :].T
#kernel_test = np.dot(dtm_test, b )
#kernel_test = dtm_test@b
#kernel_test = np.dot(dtm_test, dtm_train.T)
#b.shape
#kernel_test.shape
#dtm_train[clf.support_, :].T.shape
#clf.support_.size
