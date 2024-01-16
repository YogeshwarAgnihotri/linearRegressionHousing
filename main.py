import lazypredict
from lazypredict.Supervised import LazyRegressor
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
print("Loading Data...")
df = pd.read_excel('Original_Data_1.xlsx')
print("Done")

# Preprocess your data here (if necessary)
# For example, handling missing values, feature encoding, etc.

# Assuming 'SalePrice' is your target column
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Encoding categorical variables (if 'object' type columns are present)
# If not, you can comment out this section
label_encoder = LabelEncoder()
encoded_X = X.apply(lambda col: label_encoder.fit_transform(col.astype(str)), axis=0, result_type='expand')

# Handle NaN values by filling them with 0
# You may choose a different strategy to handle NaNs if it makes more sense for your data
encoded_X.fillna(0, inplace=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.2, random_state=42)

# Instantiate and train a Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg.predict(X_test)

# Evaluate the model
score = r2_score(y_test, y_pred)
print("R^2 score on test set:", score)