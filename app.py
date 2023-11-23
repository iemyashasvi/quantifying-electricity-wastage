# Import necessary libraries


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import time
# Load the data
# st.set_theme('light')

def create_splash_screen():
    st.markdown(
        """
        <style>
        body {
            background-color: black;
            color: white;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
        }
        h1 {
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        h2 {
            font-size: 1.5em;
            font-weight: normal;
            margin-top: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<h1>Quantifying Electricity Wastage in Domestic Environments</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Unlocking Insights </h2>", unsafe_allow_html=True)


if 'initialized' not in st.session_state:
    create_splash_screen()

    st.session_state.initialized = True  # Mark the app as initialized
    time.sleep(3)
df1 = pd.read_csv('Household energy bill data.csv')

df = pd.read_csv('Household energy bill data.csv')




# It is not logical for a house to have -1 rooms or people, so we will change those values to 0.
# Also, some values in ave_monthly_income are negative, these will be changed to the mean of the column.
df.loc[df['num_rooms'] <= 0, 'num_rooms'] = round(df['num_rooms'].mean())
df.loc[df['num_people'] <= 0, 'num_people'] = round(df['num_people'].mean())

df.loc[df['ave_monthly_income'] < 0, 'ave_monthly_income'] = df['ave_monthly_income'].mean()

# Next, we will see if the dataset contains any duplicated rows or null values.
# The second one will be achieved by text and by visual representation showing white rectangles for null values.


# Finally, show the correlation between each column.


# Data Transformation
scale_columns = ['num_rooms', 'num_people', 'housearea', 'ave_monthly_income', 'num_children']
df[scale_columns].head()

scaler = StandardScaler()

# Scale the columns previously selected and replace them in the dataset
df[scale_columns] = scaler.fit_transform(df[scale_columns])
df.head()

# Before training the models, I will check if any of the features have low variance,
# meaning they do not change much and so they do not affect the model.
vt = VarianceThreshold()

features = df.iloc[:, :9]
vt = vt.fit_transform(features)
print('The dataset contains {} features.'.format(features.shape[1]))
print('The selector contains {} features.'.format(vt.shape[1]))

# Splitting the data into training data and testing data
x = df.iloc[:, :9]
y = df['amount_paid']

# The train set will have 80% of the data and the other 20% will be for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)


# Load the Lasso model

#check
Lasso_final = Lasso(alpha=0.5, random_state=10)

# Feature names (customize based on your dataset)
feature_names = ['num_rooms', 'num_people', 'housearea', 'ave_monthly_income', 'num_children', 'is_urban']

# Select features and target
x = df[feature_names]
y = df['amount_paid']

# Feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train the Lasso model
Lasso_final.fit(x_scaled, y)
#fin

# Streamlit app

# Sidebar for user input
st.sidebar.header("User Input")

# Create a dictionary to store user input
user_input = {}

# Collect user input for each feature
for feature in feature_names:
    user_input[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0, step=0.1)

# Prepare the user input as a DataFrame
user_df = pd.DataFrame([user_input])

# Feature scaling using the previously fitted scaler
user_df_scaled = scaler.transform(user_df[feature_names])  # Use only the relevant features

# Prediction
prediction = Lasso_final.predict(user_df_scaled)

# Display prediction
st.subheader("Prediction")
st.write(f"The predicted amount_paid is: {prediction[0]:.2f}")

# Display original data
st.subheader("Original Data")
st.dataframe(df1.head())  # Display the first few rows of the original dataset

st.subheader("Transformed Data")
st.dataframe(df.head())  # Display the first few rows of the original dataset




# Load the previously fitted scaler (assumes it's fitted on the entire dataset)
# scaler = StandardScaler()

# # Select features for scaling
# df_scaled = scaler.fit_transform(df[feature_names])

# # Use the Lasso model for prediction on the scaled user input
# prediction = Lasso_final.predict(user_df_scaled)

# # Display prediction
# st.subheader("Prediction")
# st.write(f"The predicted amount_paid is: {prediction[0]:.2f}")
# # Display original data
# st.subheader("Original Data")
# st.dataframe(df.head())  # Display the first few rows of the original dataset




# Show some graphs
st.subheader("Graphs")

# Distribution plot for 'amount_paid'
fig, ax = plt.subplots()
sns.histplot(df['amount_paid'], ax=ax)
st.pyplot(fig)

# Pair plot for selected features
selected_features = feature_names + ['amount_paid']
fig = sns.pairplot(df[selected_features])
st.pyplot(fig)

# Heatmap for correlation
fig, ax = plt.subplots()
sns.heatmap(df[selected_features].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)