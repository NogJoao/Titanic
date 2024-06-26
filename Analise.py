import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
from pydoc import help
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from IPython.display import display, HTML

train = pd.read_csv('D:/Git/TITANIC/train.csv')
train.info()
train.columns
train.describe()
train.tail()
train["Sex"].value_counts()
train["Pclass"].value_counts()
train["Survived"].value_counts()
train.dtypes

# Estuda da correlação linear entre variáveis
numericas = train.select_dtypes(include = ["int64","float64"])
matrizCorrelacao = numericas.corr()
plt.figure(figsize = (2,2))
sns.heatmap(matrizCorrelacao)

plt.boxplot(train["Fare"])

train["Fare"].plot(kind = "box")

# Matriz de gráfico de pontos
scatter_matrix(numericas, diagonal="kde")

numericas.apply(np.mean)
numericas.apply(np.std)
