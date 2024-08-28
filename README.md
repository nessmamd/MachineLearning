## Class Details


### Assignment 2 
#### Objective: Write code to perform classification and regression tasks using linear models, and describe the coding process.

#### Detailed Process:

Step 0: Libraries like numpy, pandas, and yellowbrick are imported. The yellowbrick library is installed and checked for updates.

Step 1: Loaded spam dataset using load_spam() function. The dataset’s features (X) and target (y) were inspected for size and type.

Step 2: Missing values were addressed by filling them with the most frequent value. A subset of 5% of the data was created for testing.

Step 3: Implemented LogisticRegression with different data configurations: full dataset, first two columns, and 5% data subset.

Step 4: Accuracy was computed for training and validation datasets for all model configurations.

Step 5: Results were visualized in a DataFrame summarizing data size, training accuracy, and validation accuracy.

#### Findings:

Training and validation accuracy improved with more data features and a larger dataset.
False positives (non-spam classified as spam) are considered worse as they could result in potentially harmful or unwanted emails being missed.
Note: The use of a loop for storing results in the DataFrame was suggested to be more efficient.


### Assignment 3

Objective: The assignment involves using non-linear models for classification and regression tasks, including describing the process and citing any resources used.

#### Part 1: Regression (14.5 marks)

Data Input:

Loaded concrete dataset from the yellowbrick library.
Code provided for importing the dataset.
Data Processing:

No action required as processing was completed in a previous assignment.
Implement Machine Learning Model:

Used Decision Tree, Random Forest, and Gradient Boosting Machines regression models from sklearn.
Models instantiated with max_depth = 5.
Validate Model:

Calculated average training and validation accuracies using mean squared error and R² scores with cross-validation.
Results indicated Gradient Boosting (GB) had the lowest MSE and highest R² score, showing the best performance.
Visualize Results:

Created a DataFrame to compare MSE and R² scores for Decision Tree (DT), Random Forest (RF), and Gradient Boosting (GB) models.
Questions:

Compared results with a linear model from a previous assignment, noting that tree-based models, especially GB, generally performed better.
Suggested fine-tuning hyperparameters and enhancing feature engineering to improve tree-based models.


#### Part 2: Classification 

Data Input:

Loaded wine dataset from UCI and defined column headers.
Split dataset into feature matrix X and target vector y.
Data Processing:

Inspected dataset for missing values and filled them if necessary.
Counted samples of each wine type.
Implement Machine Learning Model:

Used SVC and Decision Tree Classifier models.
Trained models on the wine dataset.
Validate Model:

Evaluated models using accuracy scores with cross-validation.
Decision Tree Classifier showed higher accuracy compared to SVC.
Visualize Results:

Created a DataFrame to compare training and validation accuracy for SVC and Decision Tree Classifier.
Generated confusion matrix and classification report for Decision Tree Classifier.

Bonus: No specific details provided.
Process Description: Code sourced from assignment instructions and scikit-learn documentation. ChatGPT was used for clarifications on code implementation.
The assignment covered a comprehensive analysis of non-linear models, focusing on their performance through various metrics and comparisons.

### Assignment 4
Objective: The assignment focuses on applying knowledge of data preprocessing, model building, and hyperparameter tuning in the context of supervised learning. It involves the Heart Disease dataset to create effective pipelines and evaluate their performance through hyperparameter tuning and stacking.

Dataset: The dataset is a subset of the Heart Disease Dataset with 14 attributes and 294 instances. Attributes include demographic, clinical, and laboratory features. The target variable, num, indicates the presence or absence of heart disease, with values from 0 (no presence) to 4 (high presence).

Tasks and Solutions:

Preprocessing Tasks:

1.1: Columns with more than 60% missing values were dropped as they provide insufficient data and could bias the model.
1.2: Missing values in numerical columns were imputed using the mean, while categorical values were imputed with the most frequent value. This method preserves the overall data distribution.
1.3: Created a ColumnTransformer to handle different types of features:
StandardScaler for numerical features
OneHotEncoder for categorical features
Passthrough for binary features
Pipeline and Modeling:

2.1: Three pipelines were created:
Logistic Regression: Simple and effective for binary classification, but limited by its linearity.
Random Forest Classifier: Handles non-linear relationships and provides feature importance but is sensitive to hyperparameters.
Support Vector Classifier (SVC): Effective for high-dimensional and complex data but less scalable.
2.2: GridSearchCV was used to find the best hyperparameters for each model:
Logistic Regression: Best F1 score of 0.767
Random Forest: Best F1 score of 0.706
SVC: Best F1 score of 0.773
2.3: A stacking classifier with Random Forest as the meta-model was employed to combine the base models. It achieved:
Mean Accuracy: 0.82 ± 0.06
Mean F1 Score: 0.74 ± 0.08
2.4: The stacking classifier generally improved performance compared to Random Forest alone but showed slightly worse performance than Logistic Regression and SVC alone. This suggests that stacking might benefit from diverse base models and careful hyperparameter tuning.
Bonus Question: Two ways to potentially improve the stacking classifier’s performance:

Introduce More Diverse Base Models: Adding models like gradient boosting or neural networks can capture different aspects of the data, improving generalization.
Perform Feature Selection: Enhance model performance by selecting or engineering features that better represent the underlying data patterns.
The assignment effectively demonstrates a comprehensive approach to handling data preprocessing, model selection, and evaluation, leading to an insightful analysis of model performance.

### Final Project 
The project involves applying Principal Component Analysis (PCA) and clustering techniques to a dataset of wheat kernels. The dataset contains geometric measurements of kernels from three wheat varieties. The project focuses on preprocessing the data, scaling it appropriately, selecting and justifying machine learning models, and optimizing hyperparameters through grid search to effectively classify the wheat varieties based on their kernel features.

