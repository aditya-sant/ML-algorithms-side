import pandas as pd
import math
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from six import StringIO
from IPython.display import Image
import pydotplus

dataset = pd.read_csv(
    r"C:\Users\Aditya\Documents\UQ\Y1\comp7703\w3\1_lebron_james_shot_chart_1_2023.csv", delimiter=',')

regrdataset = dataset[["lebron_team_score", "opponent_team_score"]]
classifDataset = dataset[["lebron_team_score", "opponent_team_score", "shot_type"]]

# Regression
XDTRegr = regrdataset
yDTRegr = regrdataset[["opponent_team_score"]]

XTrain, XTest, yTrain, yTest = train_test_split(
    XDTRegr, yDTRegr, test_size=0.3, random_state=1)

regrDTmodel = DecisionTreeRegressor(max_depth=3)

scaler = StandardScaler()
XTrain = scaler.fit_transform(XTrain)
XTest = scaler.transform(XTest)

regrDTmodel.fit(XTrain, yTrain)

testPreds = regrDTmodel.predict(XTest)

mse = mean_squared_error(yTest, testPreds)

rmse = math.sqrt(mse)
print(f"root mean squared error = {rmse}")

# plot regression decision tree
dotData = StringIO()
export_graphviz(regrDTmodel, out_file=dotData, filled=True, rounded=True,
                special_characters=True, feature_names=['Col1', 'Col2'])

graph = pydotplus.graph_from_dot_data(dotData.getvalue())
graph.write_png('NBA-Regression-Decision-Tree.png')
Image(graph.create_png())


## Classification Decision Tree

XDTclassif = classifDataset.drop(columns=["shot_type"])
yDTclassif = classifDataset["shot_type"]

XTrain, XTest, yTrain, yTest = train_test_split(
    XDTclassif, yDTclassif, test_size=0.3, random_state=1)

classifDTModel = DecisionTreeClassifier(max_depth=3)

scaler = StandardScaler()
XTrain = scaler.fit_transform(XTrain)
XTest = scaler.transform(XTest)

classifDTModel.fit(XTrain, yTrain)

treePreds = classifDTModel.predict(XTest)

print(
    f"Data loss for Decision Tree: {(1 - accuracy_score(yTest, treePreds)) * 100}")

dotData = StringIO()
export_graphviz(classifDTModel, out_file=dotData, filled=True, rounded=True,
                special_characters=True, feature_names=['Lakers', 'Opponent'],
                class_names=['True', 'False'])

graph = pydotplus.graph_from_dot_data(dotData.getvalue())
graph.write_png('NBA-Classification-Decision-Tree.png')
Image(graph.create_png())

print(XDTclassif)
print(yDTclassif)

