1. Arvore de Decisão


def f(petal_length: float, petal_width: float, sepal_length: float, sepal_width: float) -> str:
  if sepal_width > 5.0:
    if petal_width > 2.0:
      return "versicolor"
    else:
      return "virginica"
  else:
    return "setosa"


import numpy as np
import pandas as pd
import seaborn as sns

iris = sns.load_dataset("iris")
iris.head()

iris.info()

iris[['species']].drop_duplicates()

iris.query("species == 'versicolor'").describe().T

iris.query("species == 'setosa'").describe().T

iris.query("species == 'virginica'").describe().T


with sns.axes_style("whitegrid"):

  grafico = sns.scatterplot(data= iris, x="petal_length", y="sepal_length", hue="species", palette="pastel")
  grafico.set(title= "Comprimento da Pétala por comprimento da Sépala", xlabel= "Comprimento da Pétala (cm)", ylabel= "Comprimento da Sépala (cm)");
  grafico.legend_.set_title("Espécie");


with sns.axes_style("whitegrid"):

  grafico = sns.scatterplot(data=iris, x="petal_width", y="sepal_width", hue="species", palette="pastel")
  grafico.set(title= "Largura da Pétala Por Largura da Sépala", xlabel= "Largura da Pétala (cm)", ylabel= "Largura da Sépala (cm)");
  grafico.legend_.set_title("Espécie");

data = iris[["species", "sepal_length", "sepal_width", "petal_length", "petal_width"]]

data.head()

2. Treino / Teste

from sklearn.model_selection import train_test_split


predictors_train, predictors_test, target_train, target_test = train_test_split(
    data.drop("species", axis=1),
    data["species"],
    test_size=0.25,
    random_state=123
)

predictors_train.head()

predictors_train.shape

predictors_test.head()

predictors_test.shape

target_train.head()

target_train.shape

target_test.head()

target_test.shape


