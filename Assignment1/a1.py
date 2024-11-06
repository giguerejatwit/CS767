import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv('gym_members_exercise_tracking.csv')

# Feature engineering: Scale calories burned with respect to weight
data['calories_per_kg'] = data['Calories_Burned'] / data['Weight (kg)']

# Preprocess data
X = data[['Age', 'Weight (kg)', 'Avg_BPM', 'Session_Duration (hours)', 'calories_per_kg', 'Workout_Frequency (days/week)', 'BMI']]  # Features
y = data['Experience_Level']  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
