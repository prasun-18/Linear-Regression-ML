import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("Data_Set\\Housing.csv")

#print(data.to_string()) #print all the data present in data sheet 
print(data.info()) # print info of all columns and non-cloumn along with data type

x = data[["area","bedrooms","bathrooms"]] # three columns are seleted as feature of the data set (independent variable)
y = data["price"] # one column is seleted as for prediction (dependent variable)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42) # Spliting the data set for training(80%) and testing(20%)

my_model = LinearRegression() #Creating instance of linear regression model

my_model.fit(x_train,y_train) #Traing the model by fitting data set's features

prediction = my_model.predict(x_test) # Predicting the prices by using data seprated for testing, while spliting the data for training and testing

print("\nPredicted price value for testing data from data set\n")
print(prediction)

# Predicting price for manual input data depending on the features "area", "bedrooms", "bathrooms"

new_data = pd.DataFrame({
    "area":[1234],
    "bedrooms":[2],
    "bathrooms":[2]
}) #Price will be predicted for these specification area=1234, bedrooms=2, bathrooms=2 // you can change the values to see how price will vary for differnt specificaton

new_pridiction = my_model.predict(new_data)
print(f"\n\n User input/speification for the house:\n{new_data}\n")
print(f"\nPrice for the user input/speification for the house:: {new_pridiction}\n")
