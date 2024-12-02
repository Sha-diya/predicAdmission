from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras 

df=pd.read_csv(r"D:\CSE5000-Thesis\deep learning\data\Admission_Predict.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.duplicated().sum())

df.drop(columns=['Serial No.'],inplace=True)
X=df.iloc[:,0:-1]#all rows,0 to -1 all columns
y=df.iloc[:,-1]#all row, -1 columns

print(X)
print(y)
#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(X_train)
scaler=MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Build the neural network model
model = Sequential()
model.add(Dense(7, activation='relu', input_dim=7))#Hidden layer #1
model.add(Dense(7, activation='relu'))#Hidden layer #2
model.add(Dense(1, activation='linear'))#Output layer
#linear for regrission type model
print(model.summary())
#Compile the model
model.compile(loss='mean_squared_error', optimizer='Adam')
#Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)

y_pred=model.predict(X_test_scaled)
print(r2_score(y_test,y_pred))

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()




