from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load and preprocess data
df = pd.read_csv('weatherHistory.csv')
df = df.dropna()

encoder = LabelEncoder()
df['Formatted Date'] = encoder.fit_transform(df['Formatted Date'])
df['Precip Type'] = encoder.fit_transform(df['Precip Type'])
df['Summary'] = encoder.fit_transform(df['Summary'])
df['Daily Summary'] = encoder.fit_transform(df['Daily Summary'])
x = df.drop(columns=['Daily Summary'])
y = df['Daily Summary']

scale = StandardScaler()
x_scaled = scale.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42)

# Ensure data types are correct
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')

# Define the model
model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, validation_split=0.1, epochs=50, batch_size=32)

# Make predictions and evaluate
predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
