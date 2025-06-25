🌸 Iris-Data-Analysis-Summer-Training-Flames-25
🌼 Iris Dataset Analysis & KNN Classification
This project is developed as part of my Summer Training under the guidance of Ms. Gaurika Dhingra Ma’am, hosted on the Flames'25 platform. I’m proud to be a part of the Angaar Batch during this program.

The primary goal is to explore, analyze, and build a classification model using the Iris dataset and the K-Nearest Neighbors (KNN) algorithm.

📂 Dataset
The dataset contains 150 samples of iris flowers across three species, with the following features:

sepal_length (cm)

sepal_width (cm)

petal_length (cm)

petal_width (cm)

Species (Iris-setosa, Iris-versicolor, Iris-virginica)

📁 Dataset used: Iris.csv (included in this repository)

🔍 What This Project Covers
📊 Exploratory Data Analysis (EDA)
Viewed dataset structure and summary

Handled missing values and duplicate records

Explored patterns using .describe(), .info(), .nunique()

🧼 Data Preprocessing
Cleaned and renamed columns for clarity

Dropped irrelevant columns like Id

Removed outliers in sepal_width using IQR method

📈 Data Visualization
Visualized using Seaborn and Matplotlib:

Class-wise distribution

Box plots, violin plots

Histograms, scatter plots, pair plots

💡 Insight:

petal_length and petal_width are the most significant features for classification.

Iris-setosa is clearly distinguishable.

Iris-versicolor and Iris-virginica show partial overlap.

🧠 Feature Engineering
Created new features to enhance classification:

sepal_ratio = sepal_length / sepal_width

petal_ratio = petal_length / petal_width

petal_size = categorical binning (Small / Medium / Large)

💡 Insight:

Setosa has a high petal ratio and low sepal ratio, structurally distinct

Virginica generally has higher ratios than Versicolor

🏷️ Label Encoding
Encoded Species and petal_size using LabelEncoder for modeling

🤖 Model Training: K-Nearest Neighbors
Split data into train/test sets

Trained KNN models with k = 3, 5, 9

Evaluated models using accuracy and classification reports

📊 Accuracy vs K Plot
Plotted accuracy scores for k = 1 to 19

Found k = 5 yields the best result

📌 Summary of Findings
Iris-setosa is easily separable due to distinct petal dimensions

Petal length and Petal width are the strongest classification features

Feature engineering helped reduce overlap between similar species

KNN at k = 5 gave the highest accuracy

💻 Technologies Used
Python 3

Pandas, NumPy for data handling

Seaborn, Matplotlib for visualization

Scikit-learn for modeling and metrics

🧾 How to Run the Project
Clone the repository
git clone https://github.com/prakhar2402/Iris-Data-Analysis-Summer-Training-Flames-25.git
Install required libraries

pip install pandas numpy matplotlib seaborn scikit-learn
Run the script

python iris_knn_analysis.py
🎓 Training Details
Project Title: Iris Dataset Analysis and Classification using KNN

Mentor: Ms. Gaurika Dhingra Mam

Platform: Flames'25

Batch: Angaar Batch

Training Type: Summer Training Program (Data Science / ML)

Institution: Lovely Professional University (LPU)

📚 Credits
Dataset Source: UCI Iris Dataset

Guided by: Ms. Gaurika Dhingra Mam

Conducted by: Flames'25

Batch: Angaar Batch

📬 Connect With Me
🔗 LinkedIn: https://www.linkedin.com/in/prakhar-gupta-366449280/

📧 Email: prakhargupta00123456@gmail.com

⭐ If you found this project helpful, don’t forget to give it a star!
