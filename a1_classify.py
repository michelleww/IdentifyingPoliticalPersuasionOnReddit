import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  
import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.model_selection import KFold
from scipy import stats
import re
from datetime import timedelta
import time

# used as a global variable
classifiers = [SGDClassifier(), GaussianNB(), RandomForestClassifier(n_estimators=10, max_depth=5), MLPClassifier(alpha=0.05), AdaBoostClassifier()]

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    if np.sum(C) == 0:
        return 0
    else:
        return np.trace(C) / np.sum(C)

def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    sum = np.sum(C, axis=1)
    return np.divide(np.diagonal(C), sum, where=sum!=0)

def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    sum = np.sum(C, axis=0)
    return np.divide(np.diagonal(C), sum, where=sum!=0)
    

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    # use the default SGDClassifier as it is linear svm
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")
        
    best = -math.inf
    iBest = 0
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for idx in range(len(classifiers)):
            classifier = clone(classifiers[idx])
            classifier.fit(X_train, y_train)
            C = confusion_matrix(y_test, classifier.predict(X_test))
            # calculating accuacy, recall and precision
            acc = accuracy(C)
            rc = recall(C)
            pre = precision(C)
            classifier_name = re.sub('\W+', '', str(classifier.__class__).split(".")[-1])
            # For each classifier, compute results and write the following output
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rc]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in pre]}\n')
            outf.write(f'\tConfusion Matrix: \n{C}\n\n')
            # updating the current best 
            if acc > best:
                iBest = idx
                best = acc
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")
    # error checking
    if iBest >= len(classifiers):
        return (None, None)

    X_1k = None
    y_1k = None
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        best_cla = clone(classifiers[iBest])
        for size in [1000, 5000, 10000, 15000, 20000]:
            X = X_train[:size]
            y = y_train[:size]
            if size == 1000:
                X_1k = X
                y_1k = y
            best_cla.fit(X, y)
            C = confusion_matrix(y_test, best_cla.predict(X_test))
            outf.write(f'{size}: {accuracy(C):.4f}\n')
            # write the comment here to avoid overwritten by re-runing the program
        outf.write('As we can see, as the training dataset size increases, the testing accuracy also increases in a strictly increasing trend.' 
                    'Normally, We expect the testing accuracy to be increased with the increasing amount of training dataset.'
                    'We can see from size 1000-5000, the accuracy increases more than others; from 5000-20000, the datasize increases significantly, however, the accuracy only increases around 0.02 in total.' 
                    'This might be related to the capacity of the model since the capacity of the model need to be adjusted to support the increases of training data size.' 
                    'From the result of 3.1, we know that the best estimator is the AdaBoost. We only use the default value for n_estimator which is 50. It might reaches the capacity of the model.'
                    'In addition, these might realted to the features that we extract from the raw data. If the features are less relevant to the class, we would get low accuracies.')
    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")
    
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.

        # part 1
        X_32k = X_train[:32000]
        y_32k = y_train[:32000]
        # k = 5
        k_feat = 5
        selector = SelectKBest(f_classif, k_feat)
        X_new = selector.fit_transform(X_32k, y_32k)
        top_5_32k = selector.get_support(indices=True)
        p_values = selector.pvalues_
        outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')
        
        # k = 50
        k_feat = 50       
        selector2 = SelectKBest(f_classif, k_feat)
        X_new_50 = selector2.fit_transform(X_32k, y_32k)
        p_values = selector2.pvalues_
        outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')
        
        # part 2
        # fit 32k data to the best classifier
        best_cla = clone(classifiers[i])
        best_cla.fit(X_new, y_32k)
        C = confusion_matrix(y_test, best_cla.predict(selector.transform(X_test)))
        accuracy_full = accuracy(C)
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')

        # train with 1k from 3.2
        k_feat = 5
        selector_1k = SelectKBest(f_classif, k_feat)
        X_new_1k = selector_1k.fit_transform(X_1k, y_1k)
        best_cla = clone(classifiers[i])
        best_cla.fit(X_new_1k, y_1k)
        C = confusion_matrix(y_test, best_cla.predict(selector_1k.transform(X_test)))
        accuracy_1k = accuracy(C)
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')

        # part 3
        # get the top 5 indices for 1k to intersect with the top 5 for 32k
        top_5_1k = selector_1k.get_support(indices=True)
        feature_intersection = np.intersect1d(top_5_32k, top_5_1k)
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')

        # part 4
        top_5 = sorted(top_5_32k)
        outf.write(f'Top-5 at higher: {top_5}\n')
        # part 5
        outf.write('The first higher feature name is: first person pronon; liwc_i; receptiviti_emotionally_aware; receptiviti_self_conscious; receptiviti_sexual_focus.'
                    'The reason that why these features are useful is that from the features names, we can see the those features represents more about personality and education.'
                    'For example, a more aggressive and self-conscious people would use more first person pronoun. And personality and education are two of the most important factor that influence people to be more "left" or "right".'
                    'We would expect the p value to be smaller when the data size increases. As we expect the increasing accuracy with increasing data size. The smaller p value is better as it give more information about the features to help unserdatnd the feature.'
                    'The names for the 32k top 5 feature are: first person pronon; liwc_i; receptiviti_emotionally_aware; receptiviti_self_conscious; receptiviti_sexual_focus.'
                    'The reason is that these features represent more about the personality, certain class of people sometimes shares personalities. It helps to identify class better')
        


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")
    overall_kfold_accuracies = []
    # since we want to run on all initially avaliable data
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # 5 fold 
        kf = KFold(shuffle=True, n_splits=5)
        for train_index, test_index in kf.split(X_all):
            kfold_accuracies = []
            # loop through all classifiers
            for i in range(len(classifiers)):
                classifier = clone(classifiers[i])
                classifier.fit(X_all[train_index], y_all[train_index])
                C = confusion_matrix(y_all[test_index], classifier.predict(X_all[test_index]))
                acc = accuracy(C)
                kfold_accuracies.append(acc)
            overall_kfold_accuracies.append(kfold_accuracies)
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')

        # get p value by comparing the best with all other classifiers
        overall_kfold_accuracies = np.array(overall_kfold_accuracies)
        p_values = []
        for idx in range(len(classifiers)):
            if idx != i:
                # overall_kfold_accuracies[idx] is the list of accuracies for the classifier at idx accross 5 folds
                S = stats.ttest_rel(overall_kfold_accuracies[:,idx], overall_kfold_accuracies[:,i])
                p_values.append(S.pvalue)  
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
        

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    data = np.load(args.input)
    data = data[data.files[0]]
    best_accuracy =[]
    X_train, X_test, y_train, y_test = train_test_split(data[:, :173], data[:, -1], 
            test_size=0.2,random_state=2, stratify=data[:, -1])
    X_train, y_train = shuffle(X_train, y_train)
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    elapsed = (time.time() - start)
    print('Finish processing for task 3.1 - 3.4 at: ' + str(timedelta(seconds=elapsed)) + ' mins')

