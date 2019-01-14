# data preprocessing

# loading libraries

import numpy as np
import pandas as pd
import matplotlib as mplt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import jaccard_similarity_score, f1_score

cell_df = pd.read_csv("cell_samples.csv")

print(cell_df.describe())
print(cell_df.head())

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
x = np.asarray(feature_df)
print(x[0:5])

cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
print(y[0:5])

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
print ('Train set data:', x_train,  y_train)
print ('Test set data:', x_test ,  y_test)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


svm_clf = svm.SVC(kernel="rbf")
svm_clf.fit(x_train,y_train)

yhat = svm_clf.predict(x_test)
print(yhat[0:5])

#This function prints and plots the confusion matrix.
def plot_confusion_matrix(cm, classes,normalize=True,title='Confusion matrix', cmap=pyplot.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    pyplot.title(title)
    pyplot.colorbar()
    tick_marks = np.arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation=45)
    pyplot.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')
    pyplot.show()

#Accuracy score using Jaccard and F1 score
print("Jaccard score accuracy",jaccard_similarity_score(y_test, yhat))

print("F1 score accuracy",f1_score(y_test, yhat,average='weighted'))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

#Classification report
print ("Classification report", classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
mplt.pyplot.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= True,  title='Confusion matrix')

