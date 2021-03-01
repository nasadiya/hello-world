#testing git fetch, pull, (push)

import json
import os
import numpy as np
import pandas as pd
from matplotlib.pyplot import plot

config = json.load(open(".\\venv\\config.json"))

data_train = pd.read_csv(config["base"]["data_path"] + config["base"]["data_file_train"])


data_train.loc[data_train.Embarked == 'S',"Embarked_num"] = 1
data_train.loc[data_train.Embarked == 'C',"Embarked_num"] = 2
data_train.loc[data_train.Embarked == 'Q',"Embarked_num"] = 3

data_train.loc[data_train.Sex == 'male',"Sex_num"] = 0
data_train.loc[data_train.Sex == 'female',"Sex_num"] = 1

data_train["Age_num"]: int = 0
for i in range(data_train.shape[0]):
    if data_train.loc[i,"Age"] > 0 :
        data_train.loc[i,"Age_num"] = int(data_train.loc[i,"Age"]/10)

# write the code for Least Squares , LASSO etc.
#data_train.Survived.hist(by=[data_train.Age_num, data_train.Sex])

data_train_reduced = data_train[["Survived", "Pclass", "Sex", "Age","SibSp", "Parch",
                                "Fare","Embarked","Embarked_num", "Sex_num", "Age_num"]]

data_train_reduced = data_train_reduced[data_train_reduced.Age >= 0]
data_train_reduced = data_train_reduced[data_train_reduced.Embarked_num >= 0]
data_train_reduced = data_train_reduced.dropna()


#list_of_covariates = ["Pclass", "Sex_num", "Age"]
list_of_covariates = ["Pclass", "Sex_num", "Age", "SibSp", "Parch", "Fare", "Embarked_num"]
y = np.array(data_train_reduced.Survived)
x = np.array(data_train_reduced[list_of_covariates])

lamb = 2
l = lasso_grad(y,x,lamb)
o = ols_estimate(y,x)
dist(y,x,l)
dist(y,x,o)
l
o
l = np.ndarray(shape=(100,1))
for i in range(100):
    l[i] = dist(y,x,lasso_grad(y,x,(i/100)))



plot(l, color='green')
plt.xlabel("Energy Source")
plt.ylabel("Energy Output (GJ)")
plt.title("Energy output from various fuel sources")
