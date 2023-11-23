# Smart House Predictor App

![App Screenshot](/screenshot.png)

## Overview

The Smart House Predictor App is a web application built using Streamlit that predicts household energy bill amounts based on user input. The app utilizes a Lasso regression model trained on a dataset containing various features related to household characteristics. 

The**Visualisation and analytics.ipynb** file contains the jupyter notebook that contains 3 models training and all the visualisation.

The app.py is for running the model and taking in user inputs and predict.


## Features

- **Prediction**: Users can input information about their house, such as the number of rooms, number of people, house area, average monthly income, number of children, and whether the house is in an urban area. The app then predicts the amount of the household's energy bill.

- **Visualization**: The app provides visualizations of the original dataset, including distribution plots, pair plots, and a heatmap of feature correlations.

## Installation

1. Clone the repository:

   ``` git clone https://github.com/iemyashasvi/quantifying-electricity-wastage ```

 

2. Run the **app.py** file
 ``` streamlit run app.py ```
 