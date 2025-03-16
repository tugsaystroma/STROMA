import streamlit as st
import altair as alt
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import keras_tuner as kt
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import KFold
import datetime as dt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

# Configure the logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s', 
    level=logging.INFO,  # You can change this to DEBUG or WARNING based on your need
    handlers=[
        logging.StreamHandler()  # Logs to the console (visible in Streamlit logs)
    ]
)

### STREAMLIT APPLICATION
st.set_page_config(page_title="STROMA", page_icon="📉", layout="wide")
# CSS to style specific headings and hide the GitHub and menu toolbar
custom_css = """
    <style>
        h3#forecasted-glucose-levels {
            color: #004AAD;  /* Blue color for specific heading */
        }
        h4#simulation-values-tracker {
            margin-top: 30px;  /* Additional top margin for another heading */
        }
        div[data-testid="stToolbarActions"] {
            display: none !important;  /* Hide the toolbar actions */
        }
        ._profileContainer_gzau3_53 {
            display: none !important;  /* Hide creator profile container */
        }
        ._container_gzau3_1 {
            display: none !important;  /* Hide Hosted by Streamlit link container */
        }
    </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)

### DEFINE FUNCTIONS, SESSION STATES, AND CONSTANTS
script_dir = os.path.dirname(os.path.abspath(__file__))

def get_food_data():
    # Load the CSV file (food data)
    csv_path = os.path.join(script_dir, 'food_data.csv')
    food_df = pd.read_csv(csv_path)

    # Create a dictionary where food names are the keys (labels) and the carbohydrate/fat amounts are the values
    food_dict_carbs = dict(zip(food_df['Food'], food_df['Amount of carbohydrates per 100 grams']))
    food_dict_fat = dict(zip(food_df['Food'], food_df['Amount of fats per 100 grams']))
    return food_dict_carbs, food_dict_fat

# FOOD DICTIONARY
food_dict_carbs, food_dict_fat = get_food_data()

# INSULIN TYPES
insulin_types = ['Fast', 'Intermediate', 'Long']

if 'is_running' not in st.session_state:
    st.session_state.is_running = False

if 'user_input_status' not in st.session_state:
    st.session_state.user_input_status = "USER INPUT STATUS: (no user input yet)"

if 'selected_food' not in st.session_state:
    st.session_state.selected_food = sorted(list(food_dict_carbs.keys()))[0]

# Store the food amount in session state to persist the input
if 'amount_in_grams' not in st.session_state:
    st.session_state.amount_in_grams = 0.0

if 'carbohydrates' not in st.session_state:
    st.session_state.carbohydrates = 0.0

# Initialize session states for data point counter and cumulative data
if 'data_point_counter' not in st.session_state:
    st.session_state.data_point_counter = 0

if 'cumulative_added_data' not in st.session_state:
    st.session_state.cumulative_added_data = pd.DataFrame()

## USER INPUT
# Initialize session states for insulin and exercise input data
if 'insulin_input_type' not in st.session_state:
    st.session_state.insulin_input_type = insulin_types[0] # Fast, Intermediate, Long

if 'insulin_input_amount' not in st.session_state:
    st.session_state.insulin_input_amount = 0.0 # In units

if 'exercise_input_kcal' not in st.session_state:
    st.session_state.exercise_input_kcal = 0.0 # In kcal

# Update the session state with the selected forecast horizon for LSTM
def set_selected_forecast_horizon_index(n):
    st.session_state.selected_forecast_horizon_index = n

# Forecast horizons for the LSTM model
forecast_horizons_list = [
    {"label": "5 MINS", "value": 1},
    {"label": "15 MINS", "value": 3},
    {"label": "30 MINS", "value": 6},
    {"label": "1 HOUR", "value": 12},
    {"label": "3 HOURS", "value": 36},
    {"label": "6 HOURS", "value": 72},
    {"label": "12 HOURS", "value": 144},
    {"label": "24 HOURS", "value": 288}
]

# Make "24 HOURS" the default selected forecast horizon
default_forecast_horizon_index = 7

## NAIVE 2
# Remove outliers using median
def remove_outliers_with_median(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = series.median()
    return np.where((series < lower_bound) | (series > upper_bound), median, series)

# Calculate the exponential moving average for Naive 2
def calculate_ema(series, span=288):
    return series.ewm(span=span, adjust=False).mean()

## XGBOOST
# Remove outliers using median
def remove_outliers_and_replace_with_median(data, threshold=2):
    if isinstance(data, pd.DataFrame):
        data_array = data.values
        columns = data.columns
    else:
        data_array = data
    
    # Calculate z-scores to detect outliers
    z_scores = np.abs(stats.zscore(data_array, axis=0))
    mask = z_scores < threshold 

    cleaned_data = np.where(mask, data_array, np.nan)
    
    for i in range(cleaned_data.shape[1]):
        if np.isnan(cleaned_data[:, i]).all():
            interval_median = 0 
        else:
            interval_median = np.nanmedian(cleaned_data[:, i])
        cleaned_data[:, i] = np.where(np.isnan(cleaned_data[:, i]), interval_median, cleaned_data[:, i])
    
    if isinstance(data, pd.DataFrame):
        cleaned_data = pd.DataFrame(cleaned_data, columns=columns, index=data.index)
    
    return cleaned_data

# Sugar decay function with exponential moving average
def sugar_decay(sugar_series, decay_rate=0.1):
    return sugar_series.ewm(span=36, adjust=False).mean() * decay_rate

## LSTM
# Define past residuals
time_step = 230

# INSULIN DECAY FUNCTION 
def calculate_dynamic_insulin_effect(insulin_df, glucose_index, decay_rates):
    """Calculate insulin effect for each time step in the prediction index."""
    insulin_effect = np.zeros(len(glucose_index))

    for insulin_type, decay_rate in decay_rates.items():
        current_insulin = insulin_df[insulin_df['Type'] == insulin_type]
        
        for idx, dose in current_insulin.iterrows():
            if idx in glucose_index:
                start_idx = glucose_index.get_loc(idx)
                steps = len(glucose_index) - start_idx
                effect = dose['Normalized Dose'] * np.exp(-decay_rate * np.arange(steps))
                insulin_effect[start_idx:start_idx + steps] += effect[:steps]

    return insulin_effect

# FAT EFFECT DECAY FUNCTION
def calculate_fat_effect(fat_df, glucose_index, fat_decay_rates):
    """
    Calculate the effect of fat intake on insulin resistance and IGL.
    """
    fat_effect = np.zeros(len(glucose_index))

    for quantile, decay_rate in fat_decay_rates.items():
        current_fat = fat_df[fat_df['Fat Intake Quantile'] == quantile]
        
        for idx, intake in current_fat.iterrows():
            if idx in glucose_index:
                start_idx = glucose_index.get_loc(idx)
                steps = len(glucose_index) - start_idx
                effect = intake['Fat Intake'] * np.exp(-decay_rate * np.arange(steps))
                fat_effect[start_idx:start_idx + steps] += effect[:steps]

    return fat_effect

# CLASSIFY FAT INTAKE INTO 20 THRESHOLDS
def classify_fat_intake_20_thresholds(amount):
    """
    Classify fat intake into 20 thresholds based on the range 0 to 50 grams.
    """
    max_fat = 50
    num_thresholds = 20
    step = max_fat / num_thresholds  

    categories = [
        'very low', 'low', 'slightly low', 'moderate-low', 
        'moderate', 'moderate-high', 'slightly high', 
        'high', 'very high', 'extremely high'
    ]

    # Map every 2 thresholds to a category (10 categories)
    thresholds = {i: categories[i // 2] for i in range(num_thresholds)}

    for i in range(num_thresholds):
        lower = i * step
        upper = (i + 1) * step
        if lower <= amount < upper:
            return thresholds[i]
    return 'out of range'  # For amounts above 50 or below 0

# NORMALIZE CARBOHYDRATE INTAKE VALUES
def normalize_carb_intake(amount, normal_threshold, max_threshold=762):
    """
    Normalize carbohydrate intake values to a range where normal_threshold is centered 
    and max_threshold is the upper bound.
    """
    if amount > max_threshold:
        return max_threshold
    return min((amount / normal_threshold) * 100, max_threshold)

# CLASSIFY CARBOHYDRATE INTAKE INTO 100 THRESHOLDS
def classify_carb_intake_100_thresholds(amount):
    """
    Classify normalized carbohydrate intake into 100 thresholds mapped to 10 categories.
    """
    step = 762 / 100  # Size of each interval for 100 thresholds
    categories = [
        'very low', 'low', 'slightly low', 'moderate-low', 
        'moderate', 'moderate-high', 'slightly high', 
        'high', 'very high', 'extremely high'
    ]
    thresholds = {i: categories[i // 10] for i in range(100)}  # Map every 10 thresholds to a category
    
    for i in range(100):
        lower = i * step
        upper = (i + 1) * step
        if lower <= amount < upper:
            return thresholds[i]
    return 'out of range'

# CARBOHYDRATE EFFECT DECAY FUNCTION
def calculate_carb_effect(carb_df, glucose_index, carb_decay_rates):
    """Calculate the effect of carbohydrate intake on glucose levels."""
    carb_effect = np.zeros(len(glucose_index))

    for quantile, decay_rate in carb_decay_rates.items():
        current_carb = carb_df[carb_df['Carb Intake Quantile'] == quantile]
        
        for idx, intake in current_carb.iterrows():
            if idx in glucose_index:
                start_idx = glucose_index.get_loc(idx)
                steps = len(glucose_index) - start_idx
                effect = intake['Normalized Carb Intake'] * np.exp(-decay_rate * np.arange(steps))
                carb_effect[start_idx:start_idx + steps] += effect[:steps]

    return carb_effect

# NORMALIZE EXERCISE VALUES TO REFLECT THEIR GLUCOSE-LOWERING EFFECT
def normalize_exercise(amount, normal_threshold, max_threshold=1000):
    """
    Normalize exercise values to a range where normal_threshold is centered 
    and max_threshold is the upper bound.
    """
    if amount > max_threshold:
        return max_threshold
    return min((amount / normal_threshold) * 100, max_threshold)

# CLASSIFY EXERCISE INTO 100 THRESHOLDS
def classify_exercise_100_thresholds(amount):
    """
    Classify normalized exercise values into 100 thresholds mapped to 10 categories.
    """
    step = 1000 / 100 
    categories = [
        'very low', 'low', 'slightly low', 'moderate-low', 
        'moderate', 'moderate-high', 'slightly high', 
        'high', 'very high', 'extremely high'
    ]
    thresholds = {i: categories[i // 10] for i in range(100)}  
    
    for i in range(100):
        lower = i * step
        upper = (i + 1) * step
        if lower <= amount < upper:
            return thresholds[i]
    return 'out of range'

# EXERCISE EFFECT DECAY FUNCTION
def calculate_exercise_effect(exercise_df, glucose_index, exercise_decay_rates):
    """Calculate the effect of exercise on glucose levels."""
    exercise_effect = np.zeros(len(glucose_index))

    for quantile, decay_rate in exercise_decay_rates.items():
        current_exercise = exercise_df[exercise_df['Exercise Quantile'] == quantile]
        
        for idx, activity in current_exercise.iterrows():
            if idx in glucose_index:
                start_idx = glucose_index.get_loc(idx)
                steps = len(glucose_index) - start_idx
                effect = activity['Normalized Exercise'] * np.exp(-decay_rate * np.arange(steps))
                exercise_effect[start_idx:start_idx + steps] += effect[:steps]

    return exercise_effect

# Adding a category based on glucose level
min_glucose_level = 70
max_glucose_level = 120

def categorize_glucose(x):
    if x < min_glucose_level:
        return 'Hypoglycemia'
    elif min_glucose_level <= x <= max_glucose_level:
        return 'Normal'
    else:
        return 'Hyperglycemia'

# Create dataset
def create_dataset(data, time_step=time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def get_cycle_timestamp(cycle_number):
    # Start time for cycle 1 (as indicated by client)
    base_time = pd.Timestamp('2020-02-29 00:03:00')

    if cycle_number == 1:
        return base_time
    else:
        # Add 5 minutes for each cycle after the first one
        minutes_to_add = (cycle_number - 1) * 5
        return base_time + pd.Timedelta(minutes=minutes_to_add)

# Add user input (food intake)
def add_food_intake(amount_in_grams):
    st.session_state.user_input_status = "USER INPUT STATUS: (has added some food intake)"

    # Calculate the intake values
    user_sugar_intake = food_dict_carbs[st.session_state.selected_food] * amount_in_grams / 100
    user_carb_intake = user_sugar_intake
    user_fat_intake = food_dict_fat[st.session_state.selected_food] * amount_in_grams / 100 
    
    # Get current timestamp in the correct format
    current_time = get_cycle_timestamp(st.session_state.data_point_counter + 1)
    date_str = current_time.strftime('%d/%m/%Y %H:%M')  # Match the existing format
    
    # Create DataFrames for new rows
    new_row_sugar = pd.DataFrame({
        'Date': [date_str],
        'SugarIntake': [user_sugar_intake]
    })
    new_row_sugar['Date'] = pd.to_datetime(new_row_sugar['Date'], format='mixed', dayfirst=True)

    new_row_carb = pd.DataFrame({
        'Datetime': [date_str],
        'Carbohydrate Intake': [user_carb_intake]
    })
    new_row_carb['Datetime'] = pd.to_datetime(new_row_carb['Datetime'], format='mixed', dayfirst=True)

    new_row_fat = pd.DataFrame({
        'Datetime': [date_str],
        'Fat Intake': [user_fat_intake]
    })
    new_row_fat['Datetime'] = pd.to_datetime(new_row_fat['Datetime'], format='mixed', dayfirst=True)

    # Ensure datetime columns in session state DataFrames are datetime type
    if 'sugar_df' in st.session_state:
        st.session_state.sugar_df['Date'] = pd.to_datetime(st.session_state.sugar_df['Date'], format='mixed', dayfirst=True)
    if 'carb_df' in st.session_state:
        st.session_state.carb_df['Datetime'] = pd.to_datetime(st.session_state.carb_df['Datetime'], format='mixed', dayfirst=True)
    if 'fat_df' in st.session_state:
        st.session_state.fat_df['Datetime'] = pd.to_datetime(st.session_state.fat_df['Datetime'], format='mixed', dayfirst=True)

    # Append new rows
    st.session_state.sugar_df = pd.concat([st.session_state.sugar_df, new_row_sugar], ignore_index=True)
    st.session_state.carb_df = pd.concat([st.session_state.carb_df, new_row_carb], ignore_index=True)
    st.session_state.fat_df = pd.concat([st.session_state.fat_df, new_row_fat], ignore_index=True)

    # Sort all DataFrames by their respective datetime columns
    st.session_state.sugar_df = st.session_state.sugar_df.sort_values(by='Date').reset_index(drop=True)
    st.session_state.carb_df = st.session_state.carb_df.sort_values(by='Datetime').reset_index(drop=True)
    st.session_state.fat_df = st.session_state.fat_df.sort_values(by='Datetime').reset_index(drop=True)

    st.success(f"Added {user_sugar_intake:.2f} grams of sugar, {user_carb_intake:.2f} grams of carbohydrates, and {user_fat_intake:.2f} grams of fat from {st.session_state.selected_food}. Will be used in the next cycle.")

def add_insulin_data(insulin_type, amount_in_units):
    st.session_state.user_input_status = "USER INPUT STATUS: (has added some insulin data)"

    # Get current timestamp in the correct format
    current_time = get_cycle_timestamp(st.session_state.data_point_counter + 1)
    date_str = current_time.strftime('%d/%m/%Y %H:%M')  # Match the existing format

    new_row_insulin = pd.DataFrame({
        'Datetime': [date_str],
        'Type': [insulin_type[0].upper()],
        'Insulin Dose': [amount_in_units]
    })
    new_row_insulin['Datetime'] = pd.to_datetime(new_row_insulin['Datetime'], format='mixed', dayfirst=True)

    # Ensure datetime column in session state DataFrame is datetime type
    if 'insulin_df' in st.session_state:
        st.session_state.insulin_df['Datetime'] = pd.to_datetime(
            st.session_state.insulin_df['Datetime'], 
            format='mixed', 
            dayfirst=True
        )

    st.session_state.insulin_df = pd.concat([st.session_state.insulin_df, new_row_insulin], ignore_index=True)
    st.session_state.insulin_df = st.session_state.insulin_df.sort_values(by='Datetime').reset_index(drop=True)

    st.success(f"Added {amount_in_units:.2f} units of {insulin_type} insulin. Will be used in the next cycle.")

def add_exercise_data(amount_in_kcal):
    st.session_state.user_input_status = "USER INPUT STATUS: (has added some exercise data)"

    # Get current timestamp in the correct format
    current_time = get_cycle_timestamp(st.session_state.data_point_counter + 1)
    date_str = current_time.strftime('%d/%m/%Y %H:%M')  # Match the existing format
    
    new_row_exercise = pd.DataFrame({
        'Datetime': [date_str],
        'Exercise (kcal)': [amount_in_kcal]
    })

    # Convert datetime column
    new_row_exercise['Datetime'] = pd.to_datetime(new_row_exercise['Datetime'], format='mixed', dayfirst=True)

    # Ensure datetime column in session state DataFrame is datetime type
    if 'exercise_df' in st.session_state:
        st.session_state.exercise_df['Datetime'] = pd.to_datetime(
            st.session_state.exercise_df['Datetime'], 
            format='mixed', 
            dayfirst=True
        )

    st.session_state.exercise_df = pd.concat([st.session_state.exercise_df, new_row_exercise], ignore_index=True)
    st.session_state.exercise_df = st.session_state.exercise_df.sort_values(by='Datetime').reset_index(drop=True)

    st.success(f"Added {amount_in_kcal:.2f} kcal of exercise. Will be used in the next cycle.")

# CACHED FUNCTIONS
@st.cache_resource
def train_xgboost(X_train, y_train, param_grid):
    # DEFINE FIXED HYPERPARAMETERS
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'colsample_bytree': 1.0
    }

    # INITIALIZE THE XGBREGRESSOR
    xgb_model = xgb.XGBRegressor(**params, n_estimators=400)

    # SET UP GRIDSEARCHCV
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                    scoring='neg_mean_squared_error', cv=3, verbose=1)

    # FIT THE MODEL
    grid_search.fit(X_train, y_train)
    # logging.info(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

@st.cache_resource
def build_lstm_model(X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(Input(shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1, clipvalue=1.0), loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

    return model

# Function to simulate glucose data updates every X minutes with forecasting
def simulate_data_addition_with_forecasting(df_cleaned, last_hour_data, interval_minutes_demo=5):
    if st.session_state.is_running:
        start_time = dt.datetime.now()  # Track start time of the simulation
        run_duration_minutes = 5  # Total simulation time (1 hour) -- CHANGE TO 60 if 1 hour, SET to 5 for demo
        elapsed_time = 0

        data_point_counter = st.session_state.data_point_counter  # Keep track of how many data points have been added
        cumulative_added_data = st.session_state.cumulative_added_data  # Keep track of cumulative added data

        st.write('<h4>Simulation Values Tracker</h4>', unsafe_allow_html=True, key="simulation_values_tracker")
        user_input_status_placeholder = st.empty() # Placeholder for testing
        cycle_counter_placeholder = st.empty() # Placeholder for testing counter


        with st.container(border=True):
            st.write('<h4>Glucose, Insulin, Exercise Data</h4>', unsafe_allow_html=True, key="glucose_data")
            col_1, col_2, col_3 = st.columns(3)
            with col_1: 
                table_title_placeholder = st.empty()  # Placeholder for the table title
                table_placeholder = st.empty()  # Placeholder for the table
            with col_2: 
                insulin_table_title_placeholder = st.empty()  # Placeholder for the insulin table title
                insulin_table_placeholder = st.empty()  # Placeholder for the insulin table
            with col_3: 
                exercise_table_title_placeholder = st.empty()  # Placeholder for the exercise table title
                exercise_table_placeholder = st.empty()  # Placeholder for the exercise table

        with st.container(border=True):
            st.write('<h4>Sugar, Fat, Carbohydrate Data</h4>', unsafe_allow_html=True, key="sugar_data")
            col_4, col_5, col_6 = st.columns(3)
            with col_4:
                sugar_table_title_placeholder = st.empty()  # Placeholder for the sugar table title
                sugar_table_placeholder = st.empty()  # Placeholder for the sugar table
            with col_5:
                fat_table_title_placeholder = st.empty()  # Placeholder for the fat table title
                fat_table_placeholder = st.empty()  # Placeholder for the fat table
            with col_6:
                carb_table_title_placeholder = st.empty()  # Placeholder for the carb table title
                carb_table_placeholder = st.empty()  # Placeholder for the carb table

        

        while elapsed_time < run_duration_minutes and st.session_state.is_running:
            user_input_status_placeholder.write(st.session_state.user_input_status)
            cycle_counter_placeholder.write(f"CYCLE COUNTER: {st.session_state.data_point_counter + 1}")


            # Add one data point from the last hour every 5 minutes
            if data_point_counter < len(last_hour_data):
                new_data_point = last_hour_data.iloc[[data_point_counter]]
                df_cleaned = pd.concat([df_cleaned, last_hour_data.iloc[[data_point_counter]]])
                cumulative_added_data = pd.concat([cumulative_added_data, new_data_point])

                # Display cumulative added data for checking
                table_title_placeholder.write("Cumulative Added Data (CGM):")
                table_placeholder.write(cumulative_added_data)
                # Display the current insulin_df
                insulin_table_title_placeholder.write("Current Insulin DataFrame:")
                insulin_table_placeholder.write(st.session_state.insulin_df)

                # Display the current exercise_df
                exercise_table_title_placeholder.write("Current Exercise DataFrame:")
                exercise_table_placeholder.write(st.session_state.exercise_df)
                
                # Display existing sugar_df for reference
                sugar_table_title_placeholder.write("Current Sugar DataFrame:")
                sugar_table_placeholder.write(st.session_state.sugar_df)

                # Display the current fat_df
                fat_table_title_placeholder.write("Current Fat DataFrame:")
                fat_table_placeholder.write(st.session_state.fat_df)

                # Display the current carb_df
                carb_table_title_placeholder.write("Current Carb DataFrame:")
                carb_table_placeholder.write(st.session_state.carb_df)



            # # Make a copy of sugar_df to avoid in-place modifications
            current_sugar_df= st.session_state.sugar_df.copy()

            # Make a copy of insulin_df to avoid in-place modifications
            current_insulin_df = st.session_state.insulin_df.copy()

            # Make a copy of fat_df to avoid in-place modifications
            current_fat_df = st.session_state.fat_df.copy()

            # Make a copy of carb_df to avoid in-place modifications
            current_carb_df = st.session_state.carb_df.copy()

            # Make a copy of exercise_df to avoid in-place modifications
            current_exercise_df = st.session_state.exercise_df.copy()

            # MEAN AND STANDARD DEVIATION OF IGL
            mean_igv = df_cleaned['Interstitial Glucose Value'].mean()
            st.session_state.mean_igv = mean_igv
            std_igv = df_cleaned['Interstitial Glucose Value'].std()
            logging.info(f"Mean of Interstitial Glucose Value: {mean_igv:.2f} mg/dL")
            logging.info(f"Standard Deviation of Interstitial Glucose Value: {std_igv:.2f} mg/dL")

            # CELL 3
            # THIS CODE IS FOR THE NAIVE 1 MODEL
            # TEST SET
            future_index = pd.date_range(df_cleaned.index[-1], periods=288 + 1, freq='5min')[1:]  
            test_data = pd.DataFrame(index=future_index)

            # TRAIN SET
            train_data = df_cleaned.copy() 

            # EXTRACT THE TIME OF DAY FROM THE INDEX FOR GROUPING
            train_data.loc[:, 'time_of_day'] = train_data.index.time
            test_data.loc[:, 'time_of_day'] = test_data.index.time

            # LOOP
            average_forecasts = {}

            for time_interval in test_data['time_of_day'].unique():
                values_at_time = train_data[train_data['time_of_day'] == time_interval]['Interstitial Glucose Value'].dropna()
                
                # CALCULATE THE AVERAGE AND IGNORE MISSING VALUES
                if len(values_at_time) > 0:
                    average_forecasts[time_interval] = values_at_time.mean()
                else:
                    average_forecasts[time_interval] = np.nan 

            # MAP THE FORECASTED VALUES TO THE TEST SET BASED ON TIME OF THE DAY
            test_data.loc[:, 'forecast'] = test_data['time_of_day'].map(average_forecasts)

            # CELL 4

            # TEST SET for NAIVE 2
            future_index = pd.date_range(df_cleaned.index[-1], periods=288 + 1, freq='5min')[1:]  
            test_data_2 = pd.DataFrame(index=future_index)

            # TRAIN SET for NAIVE 2
            train_data_2 = df_cleaned.copy()

            # EXTRACT THE TIME OF DAY FROM THE INDEX FOR GROUPING
            train_data_2.loc[:, 'time_of_day'] = train_data_2.index.time
            test_data_2.loc[:, 'time_of_day'] = test_data_2.index.time

            # COMPUTE FORECAST FOR NAIVE 2
            average_forecasts_2 = {}
            for time_interval in test_data_2['time_of_day'].unique():
                values_at_time_2 = train_data_2[train_data_2['time_of_day'] == time_interval]['Interstitial Glucose Value'].dropna()
                values_at_time_2 = pd.Series(remove_outliers_with_median(values_at_time_2))
                if len(values_at_time_2) > 0:
                    values_at_time_2_ema = calculate_ema(values_at_time_2)
                    average_forecasts_2[time_interval] = values_at_time_2_ema.mean()
                else:
                    average_forecasts_2[time_interval] = np.nan

            # APPLY THE FORECAST
            test_data_2['forecast_2'] = test_data_2['time_of_day'].map(average_forecasts_2)

            # CELL 5

            # ACTUAL AND FORECAST VALUES FOR NAIVE 1 and NAIVE 2 
            forecast_naive1 = test_data['forecast'].dropna().values
            forecast_naive2 = test_data_2['forecast_2'].dropna().values

            # ENSURE THAT THE LENGTHS MATCH
            min_len = min(len(forecast_naive1), len(forecast_naive2))
            forecast_naive1 = forecast_naive1[:min_len]
            forecast_naive2 = forecast_naive2[:min_len] 

            # CALCULATE WEIGHTS BASED ON MAPE
            weight_naive1 = 0.58
            weight_naive2 = 0.42  

            logging.info(f"Weight of NAIVE 1: {weight_naive1:.2f}")
            logging.info(f"Weight of NAIVE 2: {weight_naive2:.2f}")


            # INTEGRATION USING WEIGHTED AVERAGE APPROACH
            test_data_2['forecast_3'] = (weight_naive1 * test_data['forecast'].dropna()) + (weight_naive2 * test_data_2['forecast_2'].dropna())

            # CELL 6
            # ENDOGENOUS VARIABLE: INTERSTITIAL GLUCOSE VALUE
            df_cleaned = df_cleaned[~df_cleaned.index.duplicated(keep='first')]
            start = df_cleaned.index.min()
            end = df_cleaned.index.max()
            dates = pd.date_range(start=start, end=end, freq='5min')
            df_cleaned = df_cleaned.reindex(dates)
            df_cleaned['Interstitial Glucose Value'] = df_cleaned['Interstitial Glucose Value'].bfill()

            # TRIM THE DATA FROM THE START UNTIL IT IS DIVISIBLE BY 288
            num_elements = df_cleaned['Interstitial Glucose Value'].shape[0]
            remainder = num_elements % 288
            if remainder > 0:
                df_cleaned = df_cleaned.iloc[remainder:]

            # GROUP DATA BY DAY 
            df_cleaned['Day'] = df_cleaned.index.date
            glucose_by_day = df_cleaned.groupby('Day')['Interstitial Glucose Value'].apply(list)
            glucose_by_day = glucose_by_day.apply(lambda x: x[:288] if len(x) > 288 else x + [np.nan] * (288 - len(x)))
            glucose_array = np.array(glucose_by_day.tolist())

            # Initial dataframe with df_cleaned data
            initial_data = pd.DataFrame({
                'Time': df_cleaned.index,
                'Glucose Level': df_cleaned['Interstitial Glucose Value'],
                'Type': ['Interstitial Glucose'] * len(df_cleaned)
            })

            # Create the initial Altair chart with df_cleaned data
            initial_chart = alt.Chart(initial_data).mark_line().encode(
                x='Time:T',
                y='Glucose Level:Q',
                color=alt.Color('Type:N', scale=alt.Scale(domain=['Interstitial Glucose'], range=['#529acc']))  # Color for chart
            ).properties(
                title='Interstitial Glucose Levels (Initial Data)',
                width=700,
                height=400
            ).configure_title(
                fontSize=16,
                fontWeight='bold',
                anchor='middle',
                align='center'
            )

            with chart_section_placeholder:
                next_update_placeholder.empty()  # Clear the placeholder
                chart_title_placeholder.write('<h3>Forecasted Glucose Levels</h3>', unsafe_allow_html=True)
                with st.spinner('Processing glucose data...'):
                    st.toast("Processing glucose data...", icon="📈")
                    
                    if st.session_state.data_point_counter == 0:
                        # Display the empty chart
                        chart_placeholder.altair_chart(initial_chart, use_container_width=True)

                    # EXOGENOUS VARIABLE: CARBS/SUGAR INTAKE
                    num_nat_dates = current_sugar_df['Date'].isna().sum()
                    if num_nat_dates > 0:
                        logging.warning(f"Warning: {num_nat_dates} dates could not be parsed and are NaT. These will be handled.")
                        current_sugar_df['Date'] = current_sugar_df['Date'].ffill()

                    current_sugar_df.set_index('Date', inplace=True)
                    current_sugar_df= current_sugar_df.groupby(current_sugar_df.index).mean()
                    current_sugar_df= current_sugar_df.reindex(df_cleaned.index, method='nearest', fill_value=0)

                    cleaned_glucose_array = remove_outliers_and_replace_with_median(glucose_array)
                    daily_sugar_intake = current_sugar_df.resample('D').mean().fillna(0).values


                    # CELL 7
                    sugar_intake_lagged = current_sugar_df.shift(1)
                    sugar_intake = sugar_decay(sugar_intake_lagged).fillna(0)
                    df.loc[:, 'SugarIntake'] = sugar_intake

                    # CREATE LAGS FOR GLUCOSE
                    for lag in range(1, 13):
                        df[f'Glucose_Lag_{lag}'] = df['Interstitial Glucose Value'].shift(lag)
                    df.dropna(inplace=True)

                    # CREATE TIME-BASED FEATURES
                    df['Hour'] = df.index.hour
                    df['DayOfWeek'] = df.index.dayofweek

                    # ADD CYCLICAL TIME FEATURES
                    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
                    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
                    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
                    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

                    # SPLIT DATA
                    X = df.drop(columns=['Interstitial Glucose Value'])
                    y = df['Interstitial Glucose Value']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                    # DMATRIX FOR XGB
                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dtest = xgb.DMatrix(X_test, label=y_test)

                    # DEFINE VALUES FOR HYPERPARAMETER TUNING
                    param_grid = {
                        'eta': [0.1, 0.15, 0.2],
                        'max_depth': [8, 9, 10],
                        'subsample': [0.8, 1.0],
                    }

                    # GET THE BEST MODEL FROM GRID SEARCH
                    best_xgb_model = train_xgboost(X_train, y_train, param_grid)

                    # RECURSIVE FORECASTING: 288 PAST VALUES
                    n_steps = 288
                    last_row = df.iloc[-1].copy()
                    forecast_values = []

                    # PREDICT GLUCOSE LEVELS IN n STEPS
                    for step in range(n_steps):
                        # ENSURE THAT INPUT_FEATURES ONLY INCLUDES COLUMNS FROM THE TRAINED MODEL
                        input_features = last_row[X_train.columns].values.reshape(1, -1)  

                        # PREDICT GLUCOSE LEVELS USING THE FITTED MODEL 
                        predicted_glucose = float(best_xgb_model.predict(input_features)[0])
                        forecast_values.append(predicted_glucose)

                        # UPDATE GLUCOSE LAGS
                        for lag in range(12, 1, -1):
                            last_row[f'Glucose_Lag_{lag}'] = last_row[f'Glucose_Lag_{lag-1}']
                        
                        last_row['Glucose_Lag_1'] = predicted_glucose 

                        # UPDATE TIME FEATURES 
                        last_row['Hour'] = (last_row['Hour'] - 1) % 24  
                        if last_row['Hour'] == 23:
                            last_row['DayOfWeek'] = (last_row['DayOfWeek'] - 1) % 7

                        # UPDATE CYCLICAL TIME FEATURES
                        last_row['Hour_sin'] = np.sin(2 * np.pi * last_row['Hour'] / 24)
                        last_row['Hour_cos'] = np.cos(2 * np.pi * last_row['Hour'] / 24)
                        last_row['DayOfWeek_sin'] = np.sin(2 * np.pi * last_row['DayOfWeek'] / 7)
                        last_row['DayOfWeek_cos'] = np.cos(2 * np.pi * last_row['DayOfWeek'] / 7)

                    # GENERATE BACKWARD FORECAST FROM THE LAST AVAILABLE TIMESTAMP
                    last_timestamp = df_cleaned.index[-1]  
                    future_dates = pd.date_range(end=last_timestamp, periods=288, freq='5min')

                    # CREATE DATAFRAME FOR PREDICTED VALUES ONLY
                    forecast_df = pd.DataFrame(forecast_values[::-1], index=future_dates, columns=['Predicted Glucose'])

                    # CELL 8

                    # CALCULATE THE RESIDUALS
                    forecast_df_subset = forecast_df.copy()  
                    forecast_df_subset['Residual'] = 0  
                    residuals = forecast_df_subset['Residual'].values.reshape(-1, 1)

                    # SCALE THE RESIDUALS
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_residuals = scaler.fit_transform(residuals)

                    # CREATE SEQUENCES FOR LSTM
                    n = forecast_horizons_list[st.session_state.selected_forecast_horizon_index]["value"]  # FORECAST HORIZONS - USER INPUT

                    X, y = create_dataset(scaled_residuals, time_step)
                    X = X.reshape(X.shape[0], X.shape[1], 1)  

                    # SPLIT THE DATA INTO TRAINING AND TEST SET
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

                    history = build_lstm_model(X_train, y_train, X_val, y_val)

                    # PREDICT NEXT RESIDUALS
                    predictions = []
                    last_input = scaled_residuals[-time_step:].reshape(1, time_step, 1)

                    for _ in range(n):
                        prediction = history.predict(last_input)
                        
                        # CHECK FOR NaN OR INFINITE VALUES
                        if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
                            raise ValueError("Prediction contains NaN or infinite values.")
                        
                        predictions.append(prediction[0, 0])
                        
                        # UPDATE LAST INPUT
                        last_input = np.append(last_input[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

                    # INVERSE SCALE THE PREDICTIONS
                    predictions = np.array(predictions).reshape(-1, 1)

                    # CLIP PREDICTIONS IF NECESSARY 
                    predictions = np.clip(predictions, -1, 1)  

                    # INVERSE TRANSFORM PREDICTIONS TO ORIGINAL SCALE
                    predictions = scaler.inverse_transform(predictions)

                    # COMPARE THE RESIDUAL FORECAST WITH THE XGBOOST
                    future_forecast = forecast_df['Predicted Glucose'].values[:n] + predictions.flatten()

                    # CREATE DATA FRAME FOR FUTURE PREDICTED VALUES
                    future_index = pd.date_range(start=forecast_df.index[-1] + pd.Timedelta(minutes=5), periods=n, freq='5min')

                    # CREATE THE FUTURE DF WITH CORRECTED INDEX
                    future_df = pd.DataFrame({'Future Predicted Glucose': future_forecast}, index=future_index)

                    # OUTPUT FUTURE PREDICTIONS
                    logging.info(future_df)
                    logging.info(f"Size of future_df: {future_df.size}")

                    # CELL 10

                    # EXTRACT LSTM PREDICTED VALUES FOR THE NEXT N VALUES
                    lstm_predicted_values = future_df['Future Predicted Glucose'].values[:n] 

                    # EXTRACT COMBINED FORECAST FOR THE SAME N VALUES
                    combined_forecast_1 = test_data_2['forecast_3'].values[:n]  

                    # MEAN WEIGHTS
                    weight_lstm = 0.32 
                    weight_combined = 0.68 

                    logging.info(f"lstm weight: {weight_lstm:.2f}%")
                    logging.info(f"naive weight: {weight_combined:.2f}%")

                    # INTEGRATION USING WEIGHTED AVERAGE APPROACH
                    new_combined_forecast = (weight_lstm * lstm_predicted_values) + (weight_combined * combined_forecast_1)

                    # CREATE A NEW DATAFRAME FOR THE NEW COMBINED FORECAST
                    future_index = pd.date_range(start=forecast_df.index[-1], periods=n, freq='5min')
                    new_combined_df = pd.DataFrame({'New Combined Forecast': new_combined_forecast}, index=future_index)

                    # CELL 12 (INSULIN EFFECT)
                    # CLEAN INSULIN DATA
                    current_insulin_df = current_insulin_df.dropna()
                    current_insulin_df['Datetime'] = pd.to_datetime(current_insulin_df['Datetime'], format='mixed', dayfirst=True, errors='coerce')
                    current_insulin_df = current_insulin_df.set_index('Datetime')  # Set datetime as index

                    # NORMALIZE INSULIN DOSES TO REFLECT THEIR GLUCOSE-LOWERING EFFECT
                    dose_scaling_factor = std_igv 
                    current_insulin_df['Normalized Dose'] = current_insulin_df['Insulin Dose'] / dose_scaling_factor

                    # DECAY TIME SPANS IN HOURS 
                    time_spans = {'F': 2.5, 'I': 8, 'L': 5} 
                    interval_minutes = 5

                    # CONVERT DECAY RATES BASED ON 5-MINUTES TIME INTERVALS 
                    decay_rates = {
                        insulin_type: np.log(2) / ((time_span * 60) / interval_minutes)
                        for insulin_type, time_span in time_spans.items()
                    }

                    # COMBINE THE INSULIN EFFECT WITH THE FINAL PREDICTION 
                    insulin_effect = calculate_dynamic_insulin_effect(current_insulin_df, new_combined_df.index, decay_rates)
                
                    # Create a copy to avoid modifying the original
                    new_combined_df['Adjusted Prediction (with Insulin Effect)'] = new_combined_df['New Combined Forecast'].copy()
                    new_combined_df['Adjusted Prediction (with Insulin Effect)'] = new_combined_df['Adjusted Prediction (with Insulin Effect)'] - insulin_effect
                    new_combined_df['Adjusted Prediction (with Insulin Effect)'] = new_combined_df['Adjusted Prediction (with Insulin Effect)'].clip(lower=0)

                    # CELL 13 (FAT EFFECT)
                    # CLEAN FAT INTAKE DATA
                    current_fat_df = current_fat_df.dropna()
                    current_fat_df['Datetime'] = pd.to_datetime(current_fat_df['Datetime'], format='mixed', dayfirst=True, errors='coerce')
                    current_fat_df = current_fat_df.set_index('Datetime')  # Set datetime as index

                    current_fat_df['Fat Intake Quantile'] = current_fat_df['Fat Intake'].apply(classify_fat_intake_20_thresholds)

                    # FAT EFFECT DURATION IN HOURS (UPDATED FOR EACH QUANTILE)
                    fat_effect_base_hours = {
                        'very low': 1.0, 'low': 1.5, 'slightly low': 2.0, 'moderate-low': 2.5,
                        'moderate': 3.0, 'moderate-high': 3.5, 'slightly high': 4.0, 
                        'high': 4.5, 'very high': 5.0, 'extremely high': 5.5
                    }
                    interval_minutes = 5

                    # CONVERT FAT EFFECT DECAY RATES BASED ON 5-MINUTE INTERVALS
                    fat_decay_rates = {
                        quantile: np.log(2) / ((hours * 60) / interval_minutes)
                        for quantile, hours in fat_effect_base_hours.items()
                    }

                    # CALCULATE FAT EFFECT
                    fat_effect = calculate_fat_effect(current_fat_df, new_combined_df.index, fat_decay_rates)

                    # COMBINE THE FAT EFFECT WITH THE FINAL PREDICTION
                    # Create a copy to avoid modifying the original
                    new_combined_df['Adjusted Prediction (with Insulin + Fat Effects)'] = (
                        new_combined_df['Adjusted Prediction (with Insulin Effect)'].copy() + fat_effect
                    )
                    new_combined_df['Adjusted Prediction (with Insulin + Fat Effects)'] = new_combined_df['Adjusted Prediction (with Insulin + Fat Effects)'].clip(lower=0)


                    # CELL 14 (CARBS EFFECT)
                    # CLEAN CARBOHYDRATE INTAKE DATA
                    current_carb_df = current_carb_df.dropna()
                    current_carb_df['Datetime'] = pd.to_datetime(current_carb_df['Datetime'], format='mixed', dayfirst=True, errors='coerce')
                    current_carb_df = current_carb_df.set_index('Datetime')  # Set datetime as index

                    current_carb_df['Normalized Carb Intake'] = current_carb_df['Carbohydrate Intake'].apply(
                        lambda x: normalize_carb_intake(x, st.session_state.mean_igv * 2)
                    )

                    current_carb_df['Carb Intake Quantile'] = current_carb_df['Normalized Carb Intake'].apply(classify_carb_intake_100_thresholds)

                    # CARBOHYDRATE EFFECT DURATION IN HOURS (MAPPED TO 10 CATEGORIES)
                    carb_effect_base_hours = {
                        'very low': 1.0,
                        'low': 1.5,
                        'slightly low': 2.0,
                        'moderate-low': 2.5,
                        'moderate': 3.0,
                        'moderate-high': 3.5,
                        'slightly high': 4.0,
                        'high': 4.5,
                        'very high': 5.0,
                        'extremely high': 5.5
                    }
                    interval_minutes = 5

                    # CONVERT CARBOHYDRATE EFFECT DECAY RATES BASED ON 5-MINUTE INTERVALS
                    carb_decay_rates = {
                        quantile: np.log(2) / ((hours * 60) / interval_minutes)
                        for quantile, hours in carb_effect_base_hours.items()
                    }

                    # CALCULATE CARBOHYDRATE EFFECT
                    carb_effect = calculate_carb_effect(current_carb_df, new_combined_df.index, carb_decay_rates)

                    # COMBINE THE CARBOHYDRATE EFFECT WITH THE FINAL PREDICTION
                    # Create a copy to avoid modifying the original
                    new_combined_df['Adjusted Prediction (with Insulin + Fat + Carb Effects)'] = (
                        new_combined_df['Adjusted Prediction (with Insulin + Fat Effects)'].copy() + carb_effect
                    )
                    new_combined_df['Adjusted Prediction (with Insulin + Fat + Carb Effects)'] = new_combined_df['Adjusted Prediction (with Insulin + Fat + Carb Effects)'].clip(lower=0)


                    # CELL 15 (EXERCISE EFFECT)
                    # CLEAN EXERCISE DATA
                    current_exercise_df = current_exercise_df.dropna()
                    current_exercise_df['Datetime'] = pd.to_datetime(current_exercise_df['Datetime'], format='mixed', dayfirst=True, errors='coerce')
                    current_exercise_df = current_exercise_df.set_index('Datetime')  # Set datetime as index

                    current_exercise_df['Normalized Exercise'] = current_exercise_df['Exercise (kcal)'].apply(lambda x: normalize_exercise(x, st.session_state.mean_igv * 10))

                    current_exercise_df['Exercise Quantile'] = current_exercise_df['Normalized Exercise'].apply(classify_exercise_100_thresholds)

                    # EXERCISE EFFECT DURATION IN HOURS (MAPPED TO 10 CATEGORIES)
                    exercise_effect_base_hours = {
                        'very low': 2.0,
                        'low': 3.0,
                        'slightly low': 4.0,
                        'moderate-low': 5.0,
                        'moderate': 6.0,
                        'moderate-high': 7.0,
                        'slightly high': 8.0,
                        'high': 9.0,
                        'very high': 10.0,
                        'extremely high': 11.0
                    }
                    interval_minutes = 5

                    # CONVERT EXERCISE EFFECT DECAY RATES BASED ON 5-MINUTE INTERVALS
                    exercise_decay_rates = {
                        quantile: np.log(2) / ((hours * 60) / interval_minutes)
                        for quantile, hours in exercise_effect_base_hours.items()
                    }

                    # CALCULATE EXERCISE EFFECT
                    exercise_effect = calculate_exercise_effect(current_exercise_df, new_combined_df.index, exercise_decay_rates)

                    # COMBINE THE EXERCISE EFFECT WITH THE FINAL PREDICTION
                    # Create a copy to avoid modifying the original
                    new_combined_df['Prediction (with Insulin + Fat + Carb + Exercise Effects)'] = (
                        new_combined_df['Adjusted Prediction (with Insulin + Fat + Carb Effects)'].copy() - exercise_effect
                    )
                    new_combined_df['Prediction (with Insulin + Fat + Carb + Exercise Effects)'] = new_combined_df['Prediction (with Insulin + Fat + Carb + Exercise Effects)'].clip(lower=0)

                    # CELL 11
                    # PLOT (USING ALTAIR)
                    # Combine all data into a single dataframe for Altair
                    combined_data = pd.DataFrame({
                        'Time': pd.concat([pd.Series(df_cleaned.index), pd.Series(new_combined_df.index)]).reset_index(drop=True),
                        'Glucose Level': pd.concat([df_cleaned['Interstitial Glucose Value'], new_combined_df['Prediction (with Insulin + Fat + Carb + Exercise Effects)']]).reset_index(drop=True),
                        'Type': ['Interstitial Glucose'] * len(df_cleaned) + ['Prediction (with Insulin + Fat + Carb + Exercise Effects)'] * len(new_combined_df)
                    })

                    combined_data['Glucose Category'] = combined_data['Glucose Level'].apply(categorize_glucose)

                    # Create the base chart with tooltip encoding
                    base = alt.Chart(combined_data).encode(
                        x=alt.X('Time:T', axis=alt.Axis(format='%b %d, %Y (%-I%p)', tickCount=24)),  # Show hour and minute for every tick
                        tooltip=[
                            alt.Tooltip('Time:T', title='Time', format='%b %d, %Y (%-I:%M %p)'),
                            alt.Tooltip('Glucose Level:Q', title='Glucose Level', format='.1f'),
                            alt.Tooltip('Type:N', title='Type'),
                            alt.Tooltip('Glucose Category:N', title='Status')  # Include the glucose category in the tooltip
                        ]
                    ).properties(
                        title=f'Glucose Level Forecast with Insulin, Fat, Carb, and Exercise Effects (Cycle {st.session_state.data_point_counter + 1})',
                        width=700,
                        height=700
                    )

                    # Create glucose line
                    glucose_line = base.mark_line(color='#529ACC').encode(
                        y=alt.Y('Glucose Level:Q', title='Glucose Level (mg/dL)')
                    ).transform_filter(
                        alt.datum.Type == 'Interstitial Glucose'
                    )

                    # Adjusted prediction line
                    adjusted_prediction_line = base.mark_line(color='green').encode(
                        y='Glucose Level:Q'
                    ).transform_filter(
                        alt.datum.Type == 'Prediction (with Insulin + Fat + Carb + Exercise Effects)'
                    )

                    # Points for outliers in the adjusted prediction
                    outlier_points = base.mark_point(size=50, filled=True).encode(
                        y='Glucose Level:Q',
                        color=alt.condition(
                            (alt.datum['Glucose Level'] < min_glucose_level) | (alt.datum['Glucose Level'] > max_glucose_level),
                            alt.value('red'),  # Red points for outliers
                            alt.value('transparent')  # Transparent for non-outliers
                        )
                    ).transform_filter(
                        alt.datum.Type == 'Prediction (with Insulin + Fat + Carb + Exercise Effects)'
                    )

                    # Combine the layers
                    alt_chart = (glucose_line + adjusted_prediction_line + outlier_points).encode(
                        color=alt.Color('Type:N', 
                                scale=alt.Scale(domain=['Interstitial Glucose', 'Prediction (with Insulin + Fat + Carb + Exercise Effects)'],
                                            range=['#529ACC', '#008B8B']),
                                legend=alt.Legend(
                                    orient='bottom',
                                    direction='horizontal',   
                                    titleOrient='top',
                                    padding=20,
                                    offset=0,
                                    labelLimit=500,
                                    symbolSize=500,
                                ))
                    ).properties(
                        padding={"left": 50, "right": 50, "top": 50, "bottom": 50}
                    ).configure_title(
                        fontSize=16,
                        anchor='middle',
                        align='center'
                    ).interactive()

            st.toast("Done forecasting!", icon="✅")
            # Display the final chart
            chart_placeholder.altair_chart(alt_chart, use_container_width=True)

            data_point_counter += 1  # Increment the counter after adding a data point
            st.session_state.data_point_counter = data_point_counter  # Update the session state
            st.session_state.cumulative_added_data = cumulative_added_data  # Update the session state
            
            # Sleep for 5 minutes to simulate interval (change to 5 seconds for testing)
            for remaining in range(interval_minutes_demo * 60, 0, -1):
                next_update_placeholder.info(f"Next update in {remaining // 60}:{remaining % 60:02d}")
                time.sleep(1)  # Sleep for 1 second during countdown


            # Calculate elapsed time and check if we should continue
            elapsed_time = (dt.datetime.now() - start_time).total_seconds() // 60

        # Once all points are added, display the final chart
        next_update_placeholder.success("Simulation complete! Final data processed and displayed.")
        st.session_state.is_running = False  # Stop the simulation


### CONTENT
## !TITLE
st.markdown('<h1><span style="color:#004AAD;letter-spacing: 0.15em;">STROMA</span></h1>', unsafe_allow_html=True)

## !PLACEHOLDER FOR MESSAGES
message_placeholder = st.empty()

## !FILE UPLOAD SECTION
st.write('<h4>Upload CSV Files</h4>', unsafe_allow_html=True)

with st.container(border=True):
    col_1, col_2 = st.columns(2, gap='large')
    with col_1:
        uploaded_glucose_file = st.file_uploader("**Glucose Data** CSV file", type="csv")
    with col_2:
        uploaded_sugar_file = st.file_uploader("**Sugar Data** CSV file", type="csv")

if uploaded_glucose_file and uploaded_sugar_file:
    # Read the CSV files
    try:
        # File upload success message
        message_placeholder.success("Files uploaded successfully!")

        # Define the paths to the embedded CSV files
        insulin_csv_path = os.path.join(script_dir, 'insulin_data.csv')
        fat_csv_path = os.path.join(script_dir, 'fat_intake_data.csv')
        carb_csv_path = os.path.join(script_dir, 'carbohydrate_intake_data.csv')
        exercise_csv_path = os.path.join(script_dir, 'exercise_data.csv')

        # Load the CSV files (Glucose, Sugar Intake, Insulin, Fat Intake, Carb Intake, Exercise)
        df = pd.read_csv(uploaded_glucose_file, index_col='Datetime', dtype={'Datetime': 'object'})
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M', errors='coerce')
        sugar_df = pd.read_csv(uploaded_sugar_file)
        sugar_df['Date'] = pd.to_datetime(sugar_df['Date'], format='mixed', dayfirst=True, errors='coerce') # CONVERT DATA WITH FLEXIBLE PARSING
        insulin_df = pd.read_csv(insulin_csv_path)
        fat_df = pd.read_csv(fat_csv_path)
        carb_df = pd.read_csv(carb_csv_path)
        exercise_df = pd.read_csv(exercise_csv_path)

        # CELL 2
        # CLEAN THE GLUCOSE DATA
        df_cleaned = df.dropna(subset=['Interstitial Glucose Value'])
        
        # Initialize session state for simulation
        if 'current_data' not in st.session_state:
            st.session_state.current_data = df_cleaned.iloc[:-12]  # Store initial data excluding the last hour (12 points for 5 mins interval)
            st.session_state.last_hour_data = df_cleaned.iloc[-12:]  # Last 1 hour of data

        # Initialize sugar_df in session state
        if 'sugar_df' not in st.session_state:
            # Initial sugar_df structure
            st.session_state.sugar_df = sugar_df.copy()

        # Initialize insulin_df in session state
        if 'insulin_df' not in st.session_state:
            # Initial insulin_df structure
            st.session_state.insulin_df = insulin_df.copy()

        # Initialize fat_df in session state
        if 'fat_df' not in st.session_state:
            # Initial fat_df structure
            st.session_state.fat_df = fat_df.copy()

        # Initialize carb_df in session state
        if 'carb_df' not in st.session_state:
            # Initial carb_df structure
            st.session_state.carb_df = carb_df.copy()

        # Initialize exercise_df in session state
        if 'exercise_df' not in st.session_state:
            # Initial exercise_df structure
            st.session_state.exercise_df = exercise_df.copy()

        if 'mean_igv' not in st.session_state:
            st.session_state.mean_igv = df_cleaned['Interstitial Glucose Value'].mean()

        ## !USER INPUT SECTION (MANUAL INPUT)
        st.write('<h4>User Input Section</h4>', unsafe_allow_html=True)

        m_col_1, m_col_2, m_col_3 = st.columns(3)
        with m_col_1:

            ## !USER FOOD INTAKE SECTION (USER INPUT)
            # Dropdown for food items and user input for the amount in grams
            with st.form("sugar_input_form"):  # Use a form to prevent immediate reruns
                st.write('<h4>Food Intake</h4>', unsafe_allow_html=True)

                col_1, col_2 = st.columns(2)
                with col_1:
                    selected_food = st.selectbox("Select a food item:", sorted(list(food_dict_carbs.keys())),
                                                    key="temp_selected_food",
                                                    index=sorted(list(food_dict_carbs.keys())).index(st.session_state.selected_food))

                with col_2:
                    amount_in_grams = st.number_input('Amount (grams):', min_value=0.0, max_value=1000.0, step=0.1, key="temp_amount_in_grams")
                
                submitted_food = st.form_submit_button("Add Food Intake", use_container_width=True)
        
                if submitted_food:
                    # Only update session state when form is submitted
                    st.session_state.selected_food = selected_food
                    st.session_state.amount_in_grams = amount_in_grams
                    add_food_intake(amount_in_grams)

        with m_col_2:
            ## !INSULIN DATA SECTION (USER INPUT)

            insulin_concentrations = {
                'U-100': 100,
                'U-500': 500
            }

            with st.form("insulin_input_form"):
                st.write('<h4>Insulin Data</h4>', unsafe_allow_html=True)

                col_1, col_2 = st.columns(2)
                with col_1:
                    insulin_input_type = st.selectbox("Select insulin type:", insulin_types,
                                                        key="temp_insulin_input_type",
                                                        index=insulin_types.index(st.session_state.insulin_input_type))
                with col_2:
                    selected_concentration = st.selectbox(
                        'Select insulin dose:', 
                        list(insulin_concentrations.keys()),
                        key='temp_insulin_concentration'
                    )
                
                submitted_insulin = st.form_submit_button("Add Insulin Data", use_container_width=True)

                if submitted_insulin:
                    st.session_state.insulin_input_type = insulin_input_type
                    st.session_state.insulin_input_amount = insulin_concentrations[selected_concentration]
                    add_insulin_data(insulin_input_type, insulin_concentrations[selected_concentration])

        with m_col_3:
            # !EXERCISE DATA SECTION (USER INPUT)
            with st.form("exercise_input_form"):
                st.write('<h4>Exercise Data</h4>', unsafe_allow_html=True)

                exercise_input_kcal = st.number_input('Burned Calories (kcal):', min_value=0.0, max_value=10000.0, step=1.0, key="temp_exercise_input_kcal")
                
                submitted_exercise = st.form_submit_button("Add Exercise Data", use_container_width=True)

                if submitted_exercise:
                    st.session_state.exercise_input_kcal = exercise_input_kcal
                    add_exercise_data(exercise_input_kcal)

        ## !FORECAST HORIZONS SECTION (USER INPUT)
        with st.container(border=True):
            # Set the default forecast horizon index
            if 'selected_forecast_horizon_index' not in st.session_state:
                st.session_state.selected_forecast_horizon_index = default_forecast_horizon_index
            
            st.write('<h4>Forecast Horizons</h4>', unsafe_allow_html=True)
            st.write(f"**Current forecast horizon:** {forecast_horizons_list[st.session_state.selected_forecast_horizon_index]['label']}")
            
            st.write('Select the forecast horizon for the glucose levels:')

            col_1, col_2, col_3, col_4 = st.columns(4, gap='large')

            with st.container():
                with col_1:
                    five_min_button = st.button(forecast_horizons_list[0]["label"], on_click=set_selected_forecast_horizon_index, args=(0,), use_container_width=True)
                with col_2:
                    fifteen_min_button = st.button(forecast_horizons_list[1]["label"], on_click=set_selected_forecast_horizon_index, args=(1,), use_container_width=True)
                with col_3:
                    thirty_min_button = st.button(forecast_horizons_list[2]["label"], on_click=set_selected_forecast_horizon_index, args=(2,), use_container_width=True)
                with col_4:
                    one_hour_button = st.button(forecast_horizons_list[3]["label"], on_click=set_selected_forecast_horizon_index, args=(3,), use_container_width=True)

            with st.container():
                with col_1:
                    three_hour_button = st.button(forecast_horizons_list[4]["label"], on_click=set_selected_forecast_horizon_index, args=(4,), use_container_width=True)
                with col_2:
                    six_hour_button = st.button(forecast_horizons_list[5]["label"], on_click=set_selected_forecast_horizon_index, args=(5,), use_container_width=True)
                with col_3:
                    twelve_hour_button = st.button(forecast_horizons_list[6]["label"], on_click=set_selected_forecast_horizon_index, args=(6,), use_container_width=True)
                with col_4:
                    twenty_four_hour_button = st.button(forecast_horizons_list[7]["label"], on_click=set_selected_forecast_horizon_index, args=(7,), use_container_width=True)
        

        ## !PLACEHOLDER FOR THE FORECASTED GLUCOSE LEVELS CHART
        next_update_placeholder = st.empty()  # This will hold the "Next update in..." message
        chart_section_placeholder = st.empty()
        chart_title_placeholder = st.empty()
        chart_placeholder = st.empty()

        # Run the forecast when the button is clicked
        if st.button("Forecast Glucose Levels", use_container_width=True, type='primary'):
            st.session_state.is_running = True
            # Run the simulation to add new data points every 5 minutes
            simulate_data_addition_with_forecasting(st.session_state.current_data, st.session_state.last_hour_data, interval_minutes_demo=1)


    except Exception as e:
        message_placeholder.error("Please double check that you have uploaded the right CSV files.")
        st.error(f"Error processing files: {e}",)

else:
    message_placeholder.info("Please upload all necessary CSV files.")