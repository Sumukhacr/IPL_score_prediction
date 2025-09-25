# IPL Match Score Prediction

##  Project Objective
This project aims to predict the total score of an IPL (Indian Premier League) match using a deep learning model. The model analyzes various in-match factors to provide an accurate score forecast.

---

## 🛠️ Technologies & Libraries
* **pandas**: For data manipulation and analysis.
* **numpy**: For numerical operations.
* **scikit-learn**: Used for data splitting, scaling, and evaluating model performance.
* **tensorflow & keras**: The core frameworks for building and training the neural network.
* **matplotlib & seaborn**: For data visualization and exploration.

---

## 📂 Dataset
The project uses the `ipl_data.csv` file, which contains a wide range of in-match statistics for IPL games.

---

## ⚙️ How It Works
The project follows a standard machine learning pipeline:

1.  **Data Loading & Exploration**: The `ipl_data.csv` file is loaded into a pandas DataFrame. Initial data exploration is performed to understand the structure and content.
2.  **Preprocessing**:
    * **Categorical Encoding**: Categorical features like `bat_team`, `bowl_team`, `venue`, `batsman`, and `bowler` are converted into numerical form using **Label Encoding**.
    * **Feature Scaling**: All numerical features are scaled using `MinMaxScaler` to ensure the neural network trains effectively.
    * **Data Splitting**: The data is divided into training and testing sets to evaluate the model on unseen data.
3.  **Model Building**:
    * A **Sequential Keras model** is constructed with a series of dense layers.
    * The architecture includes two hidden layers with **ReLU activation** and a final output layer with a **linear activation**.
    * The model is compiled with the **Adam optimizer** and the **Huber loss** function, which is robust to outliers.
4.  **Training & Evaluation**:
    * The model is trained on the preprocessed training data over a specified number of epochs.
    * Predictions are made on the test set.
    * Model performance is evaluated using key regression metrics: **R-squared ($R^2$)**, **Mean Absolute Error (MAE)**, and **Mean Squared Error (MSE)**.

---

## 📊 Results & Performance
The model's accuracy is measured by the calculated R-squared, MAE, and MSE scores. A higher R-squared score and lower MAE and MSE values indicate better prediction accuracy.
