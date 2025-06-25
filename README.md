# Iris-Data-Analysis-Summer-Training-Flames-25

🌸 Iris Dataset Analysis & KNN Classification
This project is developed as a part of my Summer Training under the guidance of Ms. Gaurika Dhingra Mam on the Flames'25 platform, as a proud member of the Angaar Batch. The goal of this project is to explore, analyze, and build a classification model using the Iris dataset and the K-Nearest Neighbors (KNN) algorithm.

📂 Dataset
The dataset consists of 150 samples of iris flowers across three species, with the following features:

sepal_length (cm)

sepal_width (cm)

petal_length (cm)

petal_width (cm)

Species (Iris-setosa, Iris-versicolor, Iris-virginica)

Dataset used: Iris.csv (included in this repository)

🔍 What This Project Covers ->
📊 Exploratory Data Analysis (EDA)
View dataset structure and summary.

Handle missing values and duplicate records.

Explore statistical patterns using .describe(), .info(), .nunique().

🧼 Data Preprocessing
Cleaned column names for simplicity.

Dropped irrelevant columns like Id.

Removed outliers from sepal_width using the IQR method.

📈 Data Visualization
Used Seaborn and Matplotlib to visualize:

Class-wise distribution

Box plots, violin plots

Histograms, scatter plots, pair plots

Insight:

Petal length and width are the most effective features in distinguishing species.

Iris-setosa is clearly distinguishable; versicolor and virginica show partial overlap.

🧠 Feature Engineering
Created new features:

sepal_ratio = sepal_length / sepal_width

petal_ratio = petal_length / petal_width

petal_size (categorized into small, medium, large using binning)

Insight:

Iris-setosa has a high petal ratio and low sepal ratio, making it structurally distinct.

Virginica usually has higher ratios than versicolor.

🏷️ Label Encoding
Encoded categorical variables (Species and petal_size) to numeric form using LabelEncoder.

🤖 Model Training: K-Nearest Neighbors
Split the dataset into training and test sets.

Trained and evaluated KNN models with k = 3, 5, and 9.

Compared performance using accuracy and classification reports.

📊 Accuracy vs K Plot
Plotted accuracy score for k = 1 to 19.

Found that k = 5 provides the best accuracy.

📌 Summary of Findings
Iris-setosa is easily separable from other species.

Petal length and width are strong classification features.

Feature engineering significantly improved separation between overlapping classes.

KNN performed best at k = 5, achieving high accuracy.

💻 Technologies Used
Python 3

Pandas and NumPy for data handling

Seaborn and Matplotlib for visualization

Scikit-learn for model building and evaluation

🧾 How to Run the Project
Clone the repository:
git clone https://github.com/your-username/iris-knn-analysis.git
cd iris-knn-analysis

Install required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn
Run the Python script:
python iris_knn_analysis.py

🎓 Training Details
Project Title: Iris Dataset Analysis and Classification using KNN

Mentor: Ms. Gaurika Dhingra

Conducted by : Flames'25

Batch: Angaar Batch

Training Type: Summer Training Program (Data Science / ML)

📚 Credits
Dataset Source: Iris Dataset

Guided by: Ms. Gaurika Dhingra Mam

Conducted by: Flames'25

Batch: Angaar Batch

