import pandas as pd

# Step 1: Sample Data
data = {
    'Sleep_Hours': [6, 8, 4, 7, 5, 9, 6, 7.5, 3, 8],
    'Screen_Time': [5, 3, 7, 4, 6, 2, 5.5, 4, 8, 3],
    'Steps_Walked': [3000, 7000, 2000, 5000, 3500, 9000, 4000, 6000, 1000, 8000],
    'Stress_Level': [7, 3, 9, 5, 6, 2, 6, 4, 10, 3],
    'Water_Intake_Ltrs': [1.5, 2.5, 1, 2, 1.8, 3, 1.7, 2.2, 1, 2.5],
    'Sleep_Quality': ['Average', 'Good', 'Poor', 'Good', 'Average', 'Good', 'Average', 'Good', 'Poor', 'Good']
}

df = pd.DataFrame(data)
print(df.head())

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Encode the output
le = LabelEncoder()
df['sleep_Quality_Label'] = le.fit_transform(df['sleep_Quality_Label'])

#feature & target
x =df[['sleep_Hours', 'screen_Time', 'step_Walked', 'stress_Level', 'water_Intake_Ltrs']]
y =df['sleep_Quality_Label']


#train-test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=42)

#modeltraining
model = RandomForestClassifier()
model.fit(x_train, y_train)

#predict
y_pred = model.predict(x_test)
print("Accuracy :", accuracy_score(y_test, y_pred))