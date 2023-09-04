
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import tkinter as tk
from tkinter import messagebox
from flask import Flask, render_template_string, request

app = Flask(__name__)
# Load the dataset
data = pd.read_csv(r"C:\Users\Smile\OneDrive\Documents\smartideathon\seattle-weather.csv")

# Data preprocessing
df = data.drop(["date"], axis=1)
lc = LabelEncoder()
df["weather"] = lc.fit_transform(df["weather"])
x = df.drop("weather", axis=1)
y = df["weather"]

# Model training
xgb = XGBClassifier()
xgb.fit(x, y)

# Dictionary mapping countries to rows in the dataset
country_row_mapping = {
    "United States": 0,
    "Brazil": 1,
    "China": 2,
    "India": 3,
    "Argentina": 4,
    "Russia": 5,
    "European Union": 6,
    "Australia": 7,
    "Iran": 8,
    "Sudan": 9,
    "Pakistan": 10,
    "Nigeria": 11,
    "Bangladesh": 12,
    "Mexico": 13,
    "Indonesia": 14,
    "Vietnam": 15,
    "Turkey": 16,
    "Ukraine": 17,
    "Egypt": 18,
    "Canada": 19,
    "Pakistan": 20,
    # Add more countries here
}

# Dictionary mapping items to required temperature range
item_temp_requirements = {
    "Beef cattle": (0, 30),
    "Dairy cows": (0, 30),
    "Poultry": (5, 30),
    "Pigs": (10, 30),
    "Sheep": (5, 25),
    "Goats": (5, 30),
    "Horses": (0, 30),
    "Aquaculture (Fish)": (15, 30),
    "Eggs": (5, 30),
    "Honey": (10, 30),
    "Grains": (5, 30),
    "Fruits": (5, 40),
    "Vegetables": (10, 35),
    "Oilseeds": (5, 35),
    "Tree crops (Coffee, Cocoa, Tea, Nuts)": (15, 35),
    "Timber": (0, 35),
    "Cotton": (10, 35),
    "Sugar crops": (15, 40),
    "Spices": (15, 35),
    "Medicinal Plants and Herbs": (10, 35),
    # Add more items here
}

# Dictionary mapping countries to their vegetation
country_vegetation = {
    "United States": ["Grains", "Fruits", "Vegetables", "Timber", "Cotton"],
    "Brazil": ["Dairy cows", "Poultry", "Tree crops (Coffee, Cocoa, Tea, Nuts)", "Sugar crops"],
    "China": ["Pigs", "Sheep", "Goats", "Aquaculture (Fish)", "Cotton", "Spices"],
    "India": ["Dairy cows", "Poultry", "Aquaculture (Fish)", "Eggs", "Honey", "Grains", "Fruits", "Vegetables", "Oilseeds", "Spices", "Medicinal Plants and Herbs"],
    "Argentina": ["Beef cattle"],
    "Russia": ["Grains", "Oilseeds", "Timber"],
    "European Union": ["Pigs"],
    "Australia": ["Sheep"],
    "Iran": ["Sheep"],
    "Sudan": ["Sheep"],
    "Pakistan": ["Goats", "Cotton"],
    "Nigeria": ["Goats"],
    "Bangladesh": ["Goats"],
    "Mexico": ["Horses"],
    "Indonesia": ["Aquaculture (Fish)", "Tree crops (Coffee, Cocoa, Tea, Nuts)"],
    "Vietnam": ["Aquaculture (Fish)", "Tree crops (Coffee, Cocoa, Tea, Nuts)"],
    "Turkey": ["Honey", "Vegetables"],
    "Ukraine": ["Grains"],
    "Egypt": ["Vegetables"],
    "Canada": ["Timber"],
    "Canada": ["Pigs"],
    # Add more countries and vegetation here
}

def index():
    if request.method == "POST":
        user_country = request.form.get("country")
        user_row = country_row_mapping.get(user_country, None)

        if user_row is None:
            return render_template("index.html", result="Country not found in the dataset.")

        user_input = x.iloc[user_row, :].values.reshape(1, -1)
        predicted_weather_number = xgb.predict(user_input)[0]

        total_yield_increase = 0
        yield_increase_per_item = {}

        for item, temp_range in item_temp_requirements.items():
            if item in country_vegetation[user_country]:
                min_temp, max_temp = temp_range
                current_temp = np.random.randint(min_temp, max_temp + 1)

                yield_increase = 0
                if min_temp <= current_temp <= max_temp:
                    yield_increase = 1

                total_yield_increase += yield_increase
                yield_increase_per_item[item] = yield_increase

        if total_yield_increase > 0:
            result = "Increase in global economy."
        else:
            result = "No significant change in the global economy."

        return render_template("index.html", result=result)

    return render_template("index.html", result="")

if __name__ == "__main__":
    app.run(debug=True)