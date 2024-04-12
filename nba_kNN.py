import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

data = pd.read_csv(
    r"C:\Users\Aditya\Documents\UQ\Y1\comp7703\w3\2_james_harden_shot_chart_2023.csv", delimiter=',')


# Regression
def nbaRegr(dataset):
    regrdataset = dataset[["top", "left"]]
    regrdataset = regrdataset.fillna(0)

    regr = regrdataset.sample(frac=1).reset_index(
        drop=True)  # keep the index ordering

    XRegr = regr
    yRegr = regr.iloc[:, 1]

    regrXTrain, regrXTest, regr_yTrain, regr_yTest = train_test_split(
        XRegr, yRegr, test_size=0.3)

    print(f"X training for regression: {len(regrXTrain)}")
    print(regrXTrain)
    print(f"X testing for regression: {len(regrXTest)}")
    print(regrXTest)
    print(f"Y training for regression: {len(regr_yTrain)}")
    print(regr_yTrain)
    print(f"Y testing for regression: {len(regr_yTest)}")
    print(regr_yTest)

    regrKNNModel = KNeighborsRegressor(n_neighbors=5)


    # normalisation
    scaler = StandardScaler()
    regrXTrain = scaler.fit_transform(regrXTrain)
    regrXTest = scaler.transform(regrXTest)

    regrKNNModel.fit(regrXTrain, regr_yTrain)

    testPreds = regrKNNModel.predict(regrXTest)

    mse = mean_squared_error(regr_yTest, testPreds)

    rmse = math.sqrt(mse)
    print(f"root mean squared error = {rmse}")

    type(regrXTest) == pd.DataFrame

    # regrXtest[:, 1] could be a different sample than regr_yTest?
    plt.scatter(regrXTest[:, 0], regr_yTest, c=testPreds)
    plt.show()


# # CLASSIFICATION K-NN-------------------------------------------------------
def nbaClassif(dataset):
    classifDataset = dataset[["top", "left", "result"]]
    classif = classifDataset.sample(frac=1).reset_index(
        drop=True)  # Keep the index ordering

    x = classif.iloc[:, 0]
    y = classif.iloc[:, 1]

    training_data = list(zip(x, y))
    # print(training_data)

    classes = classif.iloc[:, 2]
    # print(classes)

    classif_X_train, classif_X_test, classif_Y_train, classif_Y_test = train_test_split(
        training_data, classes, test_size=0.3)

    print("X-axis training data:")
    print(classif_X_train)
    print("X-axis testing data:")
    print(classif_X_test)
    print("Y-axis training data:")
    print(classif_Y_train)
    print("Y-axis testing data:")
    print(classif_Y_test)

    scaler = StandardScaler()
    classif_X_train = scaler.fit_transform(classif_X_train)
    classif_X_test = scaler.transform(classif_X_test)

    k_nn = KNeighborsClassifier(n_neighbors=5)
    k_nn.fit(classif_X_train, classif_Y_train)

    classif_Y_prediction = k_nn.predict(classif_X_test)

    x_min, x_max = classif['top'].min() - 0.1, classif['top'].max() + 0.1
    y_min, y_max = classif['left'].min() - 0.1, classif['left'].max() + 0.1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    print(classif_Y_prediction)

    print(
        f"Data loss: {(1 - accuracy_score(classif_Y_test, classif_Y_prediction)) * 100}")

    plt.contourf(xx, yy, k_nn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape), cmap=plt.cm.RdYlBu , alpha=0.7)
    plt.scatter(classif.iloc[:, 0],
                classif.iloc[:, 1],
                c=classes + [classif_Y_prediction[0]], cmap=plt.cm.RdYlBu)
    plt.show()


# nbaRegr(data)
nbaClassif(data)
