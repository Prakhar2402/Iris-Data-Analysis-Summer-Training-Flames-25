# Importing required libraries for data analysis, visualization, and machine learning.
# - pandas and numpy for data manipulation
# - seaborn and matplotlib for data visualization
# - sklearn modules for preprocessing and modeling
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading the Iris dataset from CSV file.
# The dataset contains measurements of iris flowers from three species.
data = pd.read_csv("Iris.csv")
print(data)  # Glimpse of whole data

# Getting an overview of the dataset: structure, summary statistics, and shape.
# Helps identify data types, missing values, and general data distribution.
print(data.info())
print(data.describe())
print(data.head())
print(data.tail())
print(data.columns)
for item in data.columns:
    print(item)
print(data.shape)
print(data.shape[1])
print(data.shape[0])
print(data.count())
print(data['Species'])
print(data.nunique())
print(data['Species'].unique())
# Insight: The dataset contains 150 rows and 5 primary features plus an ID column and a categorical target 'Species'.
# There are three unique species: Iris-setosa, Iris-versicolor, and Iris-virginica.

# Renaming column headers for better readability and removing the ID column.
# This makes the dataset cleaner and easier to work with.
data.rename(columns={
    'SepalLengthCm': 'sepal_length',
    'SepalWidthCm': 'sepal_width',
    'PetalLengthCm': 'petal_length',
    'PetalWidthCm': 'petal_width'
}, inplace=True)
print(data.columns)
data.drop('Id', axis=1, inplace=True)
print(data.columns)
print(data.drop(1, axis=0))  # Drop row 1 (not inplace)

# Checking for missing values is crucial before proceeding with analysis.
# The dataset has no missing values, so imputation is not required here.
data.isnull()
print(data.isnull().sum())

# Checking for duplicates and removing them ensures a more accurate and efficient model.
data.duplicated()
print(data.duplicated().sum())
duplicates = data[data.duplicated()]
print(duplicates)
Duplicates = data[data.duplicated(keep=False)]
print(Duplicates)
data = data.drop_duplicates(keep='first', ignore_index=True)
print(data.duplicated().sum())
print(data.shape)
print(data.loc[:11])
print(data.value_counts('Species'))
print(data.groupby("Species").min())

# Demonstrating grouping data and calculating aggregate statistics.
data1 = {
    'Department': ['HR', 'IT', 'HR', 'Finance', 'IT'],
    'Salary': [30000, 50000, 35000, 40000, 55000]
}
df1 = pd.DataFrame(data1)
print(df1)
print(df1['Salary'].mean())
result = df1.groupby('Department')['Salary'].mean()
print(result)
print(df1.groupby('Department')['Salary'].agg(['mean', 'max', 'min']))

# Exploring the distribution of species and features using count plots and box plots.
sns.countplot(x='Species', data=data, palette="viridis")
plt.title("Species Distribution")
plt.show()
sns.countplot(x='sepal_length', data=data, palette="viridis", hue='Species')
plt.title("Species Distribution")
plt.show()
plt.figure(figsize=(10,7))
sns.boxplot(x='Species', y='sepal_length', data=data, palette="viridis")
plt.title("Boxplot of Sepal length by Species")
plt.show()
sns.boxplot(x='Species', y='sepal_width', data=data, palette="viridis")
plt.title("Boxplot of Sepal width by Species")
plt.show()
sns.boxplot(x='Species', y='petal_length', data=data ,palette='viridis')
plt.show()
sns.boxplot(x='Species', y='petal_width', data=data ,palette='viridis')
plt.show()
sns.violinplot(x='Species', y='petal_length', data=data, palette="viridis")
plt.title("Violin Plot of Petal Length by Species")
plt.show()
sns.violinplot(x='Species', y='petal_length', data=data, palette="colorblind")
plt.title("Violin Plot of Petal Length by Species")
plt.show()
sns.displot(data['sepal_length'], bins=30, kde=True)
plt.title("Distribution Plot of Sepal Length")
plt.show()
# Insights:
# - Petal length and width have significant differences across species.
# - Sepal width shows less variation and more overlapping.
# - Iris-setosa is clearly distinguishable, while versicolor and virginica overlap more.

plot = sns.FacetGrid(data, hue="Species", height=5)
plot.map(sns.histplot, "sepal_length").add_legend()
plt.show()
plot = sns.FacetGrid(data, hue="Species")
plot.map(sns.histplot, "sepal_width").add_legend()
plt.show()
plot = sns.FacetGrid(data, hue="Species")
plot.map(sns.histplot, "petal_length").add_legend()
plt.show()
plot = sns.FacetGrid(data, hue="Species")
plot.map(sns.histplot, "petal_width").add_legend()
plt.show()
fig, axes = plt.subplots(2, 2, figsize=(10,10))
axes[0,0].set_title("Sepal Length")
axes[0,0].hist(data['sepal_length'], bins=20)
axes[0,1].set_title("Sepal Width")
axes[0,1].hist(data['sepal_width'], bins=5)
axes[1,0].set_title("Petal Length")
axes[1,0].hist(data['petal_length'], bins=6)
axes[1,1].set_title("Petal Width")
axes[1,1].hist(data['petal_width'], bins=6)
axis = data.plot.hist(bins=10, alpha=0.2)
axis.set_xlabel('Size in cm')

# Visualizing feature relationships using scatter plots and pair plots.
sns.scatterplot(x='sepal_length', y='petal_length', hue='Species', data=data)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.title("Scatter Plot of Sepal Length vs. Petal Length")
plt.show()
sns.scatterplot(x='sepal_length', y='sepal_width', hue='Species', data=data)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.title("Scatter Plot of Sepal Length vs. Sepal Width")
plt.show()
sns.pairplot(data, hue='Species', height=2)
plt.show()
# Insight: Scatterplots show petal length and width are the most distinguishing features.
# Sepal features show more overlap among species, especially versicolor and virginica.

# Checking linear relationships between numeric features.
numerical_data = data.select_dtypes(include=['number'])
correlation_matrix = numerical_data.corr()
print(correlation_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()

# Using IQR method to identify and remove outliers in sepal_width.
sns.boxplot(data=data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
plt.title("Boxplot to Detect Outliers in Iris Features")
plt.show()
sns.boxplot(x='sepal_width', data=data)
Q1 = data['sepal_width'].quantile(0.25)
Q3 = data['sepal_width'].quantile(0.75)
IQR = Q3 - Q1
Lower_limit = Q1 - 1.5 * IQR
Upper_limit = Q3 + 1.5 * IQR
outliers = data[(data['sepal_width'] < Lower_limit) | (data['sepal_width'] > Upper_limit)]
print(outliers)
upper = np.where(data['sepal_width'] >= Upper_limit)
lower = np.where(data['sepal_width'] <= Lower_limit)
data.drop(upper[0], inplace=True)
data.drop(lower[0], inplace=True)
print(data.shape)

# Creating new features to improve classification.
data['sepal_ratio'] = data['sepal_length'] / data['sepal_width']
data['petal_ratio'] = data['petal_length'] / data['petal_width']
sns.scatterplot(x='sepal_ratio', y='petal_ratio', hue='Species', data=data)
plt.show()
# Insight:
# 1. Iris-setosa is well-separated from the other species - has low sepal_ratio but high petal_ratio.
#    → Narrow sepals and long, narrow petals make it structurally distinct.
# 2. Iris-versicolor and Iris-virginica overlap more, but Virginica tends to have higher ratios.
# Helped in:
# - Distinguishing Setosa (clear separation)
# - Providing some separation between Versicolor and Virginica

data['petal_size'] = pd.cut(data['petal_length'], bins=[0,2,4,7], labels=['Small','Medium','Large'])
print(data.head())
sns.countplot(x='petal_size', hue='Species', data=data)
plt.show()
data['species'] = LabelEncoder.fit_transform(data['Species'])
print(data['species'])
for i, class_label in enumerate(LabelEncoder.classes_):
    print(f"{i} → {class_label}")
data['petal_size'] = LabelEncoder.fit_transform(data['petal_size'])
print(data.head())
for i, class_label in enumerate(LabelEncoder.classes_):
    print(f"{i} → {class_label}")

# Splitting data into training and testing sets to evaluate model performance.
X = data.drop(labels=["Species", "species"], axis=1)
y = data["species"]
print(y.value_counts())
print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# KNN models trained with different values of k to compare accuracy.
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

model1 = KNeighborsClassifier(n_neighbors=5)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred1))
print(classification_report(y_test, y_pred1))

model2 = KNeighborsClassifier(n_neighbors=9)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))

# Evaluating multiple k values to find the optimal neighbor count.
accuracy_scores = []
for k in range(1, 20):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracy_scores.append(acc)

plt.plot(range(1, 20), accuracy_scores, marker='o')
plt.title("KNN Accuracy for Different k Values")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.show()
# Insight:
# - We evaluate multiple k values to find the optimal neighbor count.
# - Helps select best-performing model with maximum generalization.
