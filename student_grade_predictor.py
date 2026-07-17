import pandas as pd
from sklearn.linear_model import LinearRegression

# Student Data
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 92, 95, 98],
    "Marks": [35, 40, 45, 55, 60, 68, 75, 82, 90, 95]
}

# DataFrame
df = pd.DataFrame(data)

print("Student Data:")
print(df)

# Input and Output
X = df[["Study_Hours", "Attendance"]]
y = df["Marks"]

# Model
model = LinearRegression()
model.fit(X, y)

# User Input
hours = float(input("\nEnter Study Hours: "))
attendance = float(input("Enter Attendance (%): "))

# Prediction
new_data = pd.DataFrame({
    "Study_Hours": [hours],
    "Attendance": [attendance]
})

prediction = model.predict(new_data)

print("\nPredicted Marks:", round(prediction[0], 2))

