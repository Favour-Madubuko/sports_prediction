# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:15:49 2023

@author: Evans & Favour
"""
import numpy as np
import pickle
import streamlit as strl
import scipy.stats as stats

with open("sport_model.sav", "rb") as file: 
  scaler, loaded_model = pickle.load(file)

def predict_with_interval(Name, original_overall, input_data, model=loaded_model, scaler=scaler, confidence_level=0.95):
    # Scale the input data using the loaded scaler
  scaled_data = scaler.transform([input_data])

  # Make a point prediction using the loaded model
  point_prediction = round(model.predict(scaled_data)[0], 2)

  # Get the number of degrees of freedom (typically number of data points - 1)
  degrees_of_freedom = len(input_data) - 1

  # Calculate the standard error of the regression
  residuals = original_overall - point_prediction
  squared_residuals = residuals ** 2
  mse = np.mean(squared_residuals)
  se_regression = np.sqrt(mse)

  # Calculate the confidence interval
  alpha = 1 - confidence_level
  t_critical = stats.t.ppf(1 - alpha / 2, df=degrees_of_freedom)
  margin_of_error = t_critical * se_regression

  lower_bound = point_prediction - margin_of_error
  upper_bound = point_prediction + margin_of_error

  return f"{Name}'s overall is {round(point_prediction,2)}.\nConfidence interval is ({round(lower_bound,2)}, {round(upper_bound,2)})"

def main():
  page = strl.sidebar.selectbox("Go to", ["Home Page", "Make a Prediction"])
  
  if page=="Home Page":
    strl.title("Welcome to Evans and Favours'")
    strl.markdown("**Players Rating site**")
    strl.write("Our site has 0.775 Mean Absolute Error")

  if page=="Make a Prediction":
    # giving a title to the app
    strl.title('Potential Player Performance Prediction')
    # getting input data from the user
    Name = strl.text_input('What is the name of the player?')  
    strl.warning("Please enter only numbers from here onwards")  
    potential = strl.number_input("What is the players' potential rate", min_value=0)
    attacking_short_passing = strl.number_input("What is the players' attacking short passing rate", min_value=0)
    skill_long_passing = strl.number_input("What is the players' skill long passing rate", min_value=0)
    movement_reactions = strl.number_input("What is the players' movement reactions", min_value=0)
    power_shot_power = strl.number_input("What is the players' power shot power", min_value=0)
    mentality_composure = strl.number_input("What is the players' mentality composure", min_value=0)
    mentality_vision = strl.number_input("What is the players' mentality vision", min_value=0)
    value_eur = strl.number_input("What is theplayers' value in eur", min_value=0)
    wage_eur = strl.number_input("What is the player's wage in eur", min_value=0)
    passing = strl.number_input("What is the players' passing rate", min_value=0)
    dribbling = strl.number_input("What is the players' dribbling rate", min_value=0)    
    physic = strl.number_input("What is the players' physic rate", min_value=0)
    original_overall = strl.number_input("What is the players' original overall", min_value=0)

    # creating a button for prediction
    
    if strl.button('Predict Player Performance'):
      predicted_value=predict_with_interval(Name,original_overall,[potential,attacking_short_passing,skill_long_passing,movement_reactions,power_shot_power,mentality_vision,mentality_composure,value_eur,wage_eur,passing,dribbling,physic])

      strl.success(predicted_value)
    
    
# calling the main function to display the result    
if __name__ == '__main__':
    main()
    