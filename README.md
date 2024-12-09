# Housing Price Prediction Model

This project demonstrates the implementation of a simple housing price prediction model using **Linear Regression** from the `scikit-learn` library. The script reads housing data from a CSV file, trains the model, and evaluates its performance. It also provides functionality for predicting prices based on user-provided input specifications.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Code Walkthrough](#code-walkthrough)
- [Features](#features)
- [Evaluation Metrics](#evaluation-metrics)
- [How to Use](#how-to-use)
- [Output Examples](#output-examples)


## Prerequisites

- Python 3.7 or later
- Libraries:
  - `pandas`
  - `scikit-learn`

You can install the required libraries using the command:

```bash
pip install pandas scikit-learn
```

## Setup and Installation

1. Clone the repository or download the script.
2. Ensure the dataset file `Housing.csv` is located in the `Data_Set` directory.
3. Run the script using:

```bash
python LinearRegression.py
```


## Code Walkthrough

### Data Loading

The script reads data from `Housing.csv`:

```python
data = pd.read_csv("Data_Set\\Housing.csv")
```

### Data Preprocessing

- Features (`x`) selected: `area`, `bedrooms`, `bathrooms`.
- Target variable (`y`): `price`.

The dataset is split into training (80%) and testing (20%) subsets using:

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

### Model Training

A Linear Regression model is created and trained:

```python
my_model = LinearRegression()
my_model.fit(x_train, y_train)
```

### Prediction

The model predicts housing prices for the test data and for user-defined inputs:

```python
prediction = my_model.predict(x_test)
new_data = pd.DataFrame({"area": [1234], "bedrooms": [2], "bathrooms": [2]})
new_pridiction = my_model.predict(new_data)
```

### Evaluation

The model is evaluated using Mean Squared Error (MSE) and R² score:

```python
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)
```

## Features

- **Training and Testing**: Split data into training and testing sets.
- **Prediction**: Predict house prices based on provided features.
- **Evaluation**: Evaluate model accuracy using metrics.
- **Manual Input Prediction**: Enter custom specifications to predict prices.

## Evaluation Metrics

- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
- **R² Score**: Indicates the proportion of variance explained by the model.
- **Model Coefficients**: Slope values for the features.
- **Intercept**: The y-intercept of the regression line.

## How to Use

1. Run the script.
2. View:
   - Dataset information.
   - Predictions for the test set.
   - Predictions for user-defined inputs.
   - Model evaluation metrics.
3. Modify the `new_data` dictionary to predict prices for custom specifications.

## Output Examples

### Example Prediction

```
Predicted price value for testing data from data set:
[450000.  600000.  300000.]

User input/specificiation for the house:
   area  bedrooms  bathrooms
0  1234         2          2

Price for the user input/specification for the house: [275000.]
```

### Example Evaluation

```
Mean_Squared_Error:  1000000.0
R^2 score:  0.85
Coefficients:  [2000. 30000. 40000.]
Intercept:  50000.0
```

### Note:
```
Values may differ because, I considered random values here
```