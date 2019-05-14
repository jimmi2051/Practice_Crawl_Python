import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets,svm
import sys
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,f1_score,precision_score
from sklearn.utils.multiclass import unique_labels

import json

path_train_idf = "../output_process/Train_Document_IDF/"
path_test_idf = "../output_process/Test_Document_IDF/"

def read_file(path):
    result = []
    with open(path, 'r') as file_input:
        line = file_input.readline()
        while line:
            line_check = line.replace("\n","").strip()
            if line_check == "0.         0.         0.         0.         0.         0.":
                line = file_input.readline()
                continue
            result.append(line.replace("\n","").strip())
            line = file_input.readline()   
    return np.asarray(result)

def myweight(distances):
    sigma2 = .5 # we can change this number
    return np.exp(-distances**2/sigma2)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def practice_example():
    #Practice data example
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target
    class_names = iris.target_names
    print(str(len(iris_X)))
    print(str(len(iris_y)))
    print ('Number of classes: '+str(len(np.unique(iris_y)))) 
    print ('Number of data points: '+str(len(iris_y)) )
    print("\n")
    X0 = iris_X[iris_y == 0,:]
    print ('\nSamples from class 0:\n', X0[:5,:])

    X1 = iris_X[iris_y == 1,:]
    print('\nSamples from class 1:\n', X1[:5,:])

    X2 = iris_X[iris_y == 2,:]
    print('\nSamples from class 2:\n', X2[:5,:])
    
    X_train, X_test, y_train, y_test = train_test_split(
        iris_X, iris_y, test_size=50)



    print ("Training size: " +str(len(y_train)))
    print ("Test size    : " +str(len(y_test)))
    clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print ("Print results for 20 test data points:")
    print ("Predicted labels: ", y_pred[20:40])
    print ("Ground truth    : ", y_test[20:40])
    print ("Accuracy of 1NN: "+ str((100*accuracy_score(y_test, y_pred))))

    clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print ("Accuracy of 10NN with major voting: "+ str((100*accuracy_score(y_test, y_pred))))

    clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'distance')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print ("Accuracy of 10NN (1/distance weights): " +str((100*accuracy_score(y_test, y_pred))))

    clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print ("Accuracy of 10NN (customized weights): "+ str((100*accuracy_score(y_test, y_pred))))


    #Practice
    knn = neighbors.KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train,y_train)

    neigh = neighbors.NearestNeighbors(n_neighbors = 5)
    neigh.fit(X_train)
    (distance, found_index) = neigh.kneighbors(X_test)
    print("Distance: \n",distance)
    print("\n Index\n",found_index)
    #Accuracy
    accuracy = knn.score(X_test,y_test)
    #Confusion matrix 
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix: \n",cm)

    #Precision
    precision = precision_score(y_test,y_pred,average=None)
    print("\n Precision Score",precision)

    #Recall
    recall = recall_score(y_test,y_pred,average=None)
    print("\n Recall: ",recall)
    #F1-Score
    f1 = f1_score(y_test,y_pred,average=None)
    print("\n F1 Score:",f1)

    #Cross Value Score
    knn_cv = neighbors.KNeighborsClassifier(n_neighbors=10)
    cv_scores = cross_val_score(knn_cv,iris_X,iris_y,cv=5)
    print(cv_scores)

    mean_cv_scores = np.mean(cv_scores)
    print("cv_scores mean: {}".format(mean_cv_scores))

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=class_names,
                        title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')

    plt.show()

def read_file_2(file_path):
    result = []
    with open(file_path,'r') as file_input:
        file_text = file_input.readline()
        while file_text:
            item_arr = []
            parse_string = file_text.split(",")
            for item in parse_string:
                item_arr.append(float(item))
            result.append(item_arr)
            file_text=file_input.readline()
    return result
    # page = response.url.split("/")
def read_title(file_path):
    with open(file_path,'r') as file_input:
        file_text = file_input.read()
        parse_string = file_text.split(",")
        result = []
        for item in parse_string:
            result.append(int(item))
        return result
def main(argv):
    # practice_example()
    X = read_file_2("../output_process/TF_IDF.txt")
    print(X)
    # X = np.asarray(temp_X)
    # temp_Y = read_title("../output/label.txt")
    # Y = np.asarray(temp_Y)
    # X.reshape(X.shape[1:])
    # X = X.transpose()
    # print(X.shape)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, Y, test_size=50)



    # print ("Training size: " +str(len(y_train)))
    # print ("Test size    : " +str(len(y_test)))
    # clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    # print ("Print results for 20 test data points:")
    # print ("Predicted labels: ", y_pred[20:40])
    # print ("Ground truth    : ", y_test[20:40])
    # print ("Accuracy of 1NN: "+ str((100*accuracy_score(y_test, y_pred))))

    # clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    # print ("Accuracy of 10NN with major voting: "+ str((100*accuracy_score(y_test, y_pred))))

    # clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'distance')
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    # print ("Accuracy of 10NN (1/distance weights): " +str((100*accuracy_score(y_test, y_pred))))

    # clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    # print ("Accuracy of 10NN (customized weights): "+ str((100*accuracy_score(y_test, y_pred))))


    # #Practice
    # knn = neighbors.KNeighborsClassifier(n_neighbors=10)
    # knn.fit(X_train,y_train)

    # neigh = neighbors.NearestNeighbors(n_neighbors = 5)
    # neigh.fit(X_train)
    # (distance, found_index) = neigh.kneighbors(X_test)
    # print("Distance: \n",distance)
    # print("\n Index\n",found_index)
    # #Accuracy
    # accuracy = knn.score(X_test,y_test)
    # #Confusion matrix 
    # y_pred = knn.predict(X_test)
    # cm = confusion_matrix(y_test,y_pred)
    # print("Confusion Matrix: \n",cm)

    # #Precision
    # precision = precision_score(y_test,y_pred,average=None)
    # print("\n Precision Score",precision)

    # #Recall
    # recall = recall_score(y_test,y_pred,average=None)
    # print("\n Recall: ",recall)
    # #F1-Score
    # f1 = f1_score(y_test,y_pred,average=None)
    # print("\n F1 Score:",f1)

    # #Cross Value Score
    # knn_cv = neighbors.KNeighborsClassifier(n_neighbors=10)
    # cv_scores = cross_val_score(knn_cv,X,Y,cv=5)
    # print(cv_scores)

    # mean_cv_scores = np.mean(cv_scores)
    # print("cv_scores mean: {}".format(mean_cv_scores))

    # np.set_printoptions(precision=2)

    # # Plot non-normalized confusion matrix
    # plot_confusion_matrix(y_test, y_pred, classes=class_names,
    #                     title='Confusion matrix, without normalization')

    # # Plot normalized confusion matrix
    # plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
    #                     title='Normalized confusion matrix')

    # plt.show()
    return

if __name__ == "__main__":
    main(sys.argv[1:])