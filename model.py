import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle

df = pd.read_csv("data.csv")


df = pd.get_dummies(df, columns=["income_group"])

# independent variables
X = df.drop(["name", "malnutrition_percent"], axis=1)

# dependent variable selection
Y = df["malnutrition_percent"]

# splitting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=50
)

# scaling the data
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

# Create a base model
rf = RandomForestRegressor()

# Instantiate the grid search model
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
)

# Fit the grid search to the data
grid_search.fit(X_train, Y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the model using the best parameters
regressor = RandomForestRegressor(**best_params)
regressor.fit(X_train, Y_train)

# Save the model
pickle.dump(regressor, open("model.pkl", "wb"))
