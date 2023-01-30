# Kamil Khan

# This program analyes a dataset of employee data and creates ...
# ... a prediction model to predict if a new employee is most likely to attrite (leave)
# Uses an artificial neural network with an experimentally determined set of hidden layers
#---------------------------------------------------------------------------------------------------------------------------------

# 1 - Import libraries
import numpy as np              # Used for large and small multi-dimensional arrays and matrices, and mathematical functions
import pandas as pd             # Used for data manipulation and analysis. Use this to import data, create matrix of features and dependant variable vector
import tensorflow as tf         # Open Source library for Machine Learning and AI

# Import classes 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# 2 - Data Preprocessing ---------------------------------------------------------------------------------------------------------
# a - Read the dataset and set independent and dependent variables
dataset = pd.read_csv('Employee_Attrition_Data.csv')
x = dataset.iloc[:, 1:-2].values # These are all columns of input values. 
y = dataset.iloc[:, 0].values # Because the first column is our output values.

# b - Implement LabelEncoder to convert dependent variables with binary outcomes to binary integers
le = LabelEncoder()
y = le.fit_transform(y)
binVarsCols = [8, 14, 18, 19]
for colNum in binVarsCols:
  x[:,colNum] = le.fit_transform(x[:,colNum])

# c - Implement OneHotEncoder to convert categorical data to interpretable interger data ...
# ... Also, finds the number of unique values in each column that is OneHotEncoded. This is used to build predicitions
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1, 3, 6, 12])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

# 3 - Split Data into Training and Test Sets -------------------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

# a - Feature Scaling 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# 4 - Build the ANN -------------------------------------------------------------------------------------------------------------
# a - Initialize
ann = tf.keras.models.Sequential()
# b - Adding the input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# c - Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# d - Adding the output layer
#   - units=1 - 1 neuron needed because we only need one output
#   - sigmoid - Sigmoid gives us the probability of our outcome 
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# 5 - Training the ANN ----------------------------------------------------------------------------------------------------------
# a - Compiling the ann
#   - loss - Is the loss function. When you have a binary outcome, you HAVE to use 'binary_crossentropy'F
#          - For multiple outcomes, you have to enter 'categorical_crossentropy' and change the activation in output layer to 'soft max'
#   - metrics - You can choose different metrics to evaluate your ANN, we choose accuracy 
#   - Optimizer - Minimizes the cost function using stochastic gradient descent. One of the best ones to use is 'adam'
ann.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = 'accuracy')

# b - Training the ann with the training set
# it is more efficient to train with batch_size as opposed to each row one by one, 32 is a good size 
ann.fit(x_train, y_train, batch_size = 32, epochs = 160)
 
# 6 - Making Predictions and Evalulation ------------------------------------------------------------------------------------------
result = ann.predict(sc.transform([[0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,26,520,14,4,4,1,58,3,1,3,2,4320,9670,2,0,0,11,3,3,80,0,2,3,2,2,2,2,1]])) >= 0.5
if result == True:
    print("This employee will leave the company")
else:
    print("This employee will stay with the company")
    