import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

def main(argv):
    # x = read_file(path_train_idf+"TF-IDF.txt")
    # y = read_file(path_train_idf+"TF-IDF-index.txt")
    # print(x[1009][6])
    # x_train, x_test,y_train,y_test = train_test_split(x,y,random_state = 0,test_size=0.3)
    # knn = KNeighborsClassifier (n_neighbors=10)
    # knn.fit(x_train,y_train)

    # # neigh = NearestNeighbors(n_neighbors=5)
    # # neigh.fit(x_train)
    # # (distance, found_index) = neigh.kneighbors(x_test)
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target
    print ('Number of classes: '+str(len(np.unique(iris_y)))) 
    print ('Number of data points: '+str(len(iris_y)) )

    print(iris_X)
    print(iris_y)
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

    return

if __name__ == "__main__":
    main(sys.argv[1:])