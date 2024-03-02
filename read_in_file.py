import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

file_contents = []
for i in range(1,11):
    with open('MHEALTHDATASET/mHealth_subject'+str(i)+'.log', 'r') as file: #Read in log file
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            file_contents.append(row)

#Columns and classes gathered from mhealth readme file
columns = ['chest_acceleration_x', 'chest_acceleration_y', 'chest_acceleration_z', 'ecg_1', 'ecg_2',
           'left_ankle_acceleration_x', 'left_ankle_acceleration_y', 'left_ankle_acceleration_z',
           'left_ankle_gyro_x', 'left_ankle_gyro_y', 'left_ankle_gyro_z', 'left_ankle_magnetometer_x',
           'left_ankle_magnetometer_y', 'left_ankle_magnetometer_z', 'right_arm_acceleration_x',
           'right_arm_acceleration_y', 'right_arm_acceleration_z', 'right_arm_gyro_x', 'right_arm_gyro_y',
           'right_arm_gyro_z', 'right_arm_magnetometer_x', 'right_arm_magnetometer_y', 'right_arm_magnetometer_z',
           'class']
classes = {0: 'inbetween', 1: 'standing still', 2: 'sitting and relaxing', 3: 'lying down', 4: 'walking',
           5: 'climbing stairs', 6: 'waist bends forward', 7: 'frontal elevation of arms',
           8: 'knees bending (crouching)', 9: 'cycling', 10: 'jogging', 11: 'running', 12: 'jump front and back'}


data = pd.DataFrame(file_contents, columns=columns)
data = data[data['class'] != '0'] #Remove the in-between recordings

X = data.drop(['class'], axis=1) #Split into X and y
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier() #Train and test model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

