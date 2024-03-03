import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from pyts.classification import TimeSeriesForest
from pyts.multivariate.classification import MultivariateClassifier


def slidingWindow(df, sampling_rate, measurement):
    window_size = 5 * sampling_rate #Window size 5 seconds
    step_size = sampling_rate #Step size 1 second
    num_windows = int(len(df) / step_size)
    indexer = np.arange(window_size)[None, :] + step_size * np.arange(num_windows)[:, None] #Create indices for sliding window
    num_na = int(window_size/step_size - 1) #Remove the last few rows that will have missing values
    indexer = indexer[:-num_na]

    values = df[measurement].values
    return values[indexer] #Apply indexer to values

def transform_data(data):
    sampling_rate = 50
    X_overall = None
    y_overall = None
    for i, classification in enumerate(data['class'].unique()): #Must separate each classification
        class_subset = data[data['class'] == classification]
        X = None
        for j, measurement in enumerate(data.columns.unique()): #Loop for each feature
            if measurement != 'class':
                class_subset_window = slidingWindow(class_subset, sampling_rate, measurement)
                if j == 0:
                    X = class_subset_window
                elif j == 1:
                    X = np.stack((X, class_subset_window), axis=1)
                else:
                    X = np.concatenate((X, class_subset_window[:, np.newaxis, :]), axis=1)
        y = np.full((X.shape[0],), classification) #Create y array of class label
        if i == 0:
            X_overall = X
            y_overall = y
        else:
            X_overall = np.concatenate((X_overall, X), axis=0)
            y_overall = np.concatenate((y_overall, y), axis=0)
    return X_overall, y_overall #Return X and y numpy arrays

def dotimeseriesclassification(X_train, X_test, y_train, y_test):
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    clf = MultivariateClassifier(TimeSeriesForest(random_state=42, class_weight='balanced', n_jobs=-1))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


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

X_train = None
X_test = None
y_train = None
y_test = None
for i in range(1,11):
    file_contents = []
    with open('MHEALTHDATASET/mHealth_subject'+str(i)+'.log', 'r') as file: #Read in log file
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            file_contents.append(row)
        data = pd.DataFrame(file_contents, columns=columns)
        data = data.astype(float)
        data['class'] = data['class'].astype(int)

        data = data[data['class'] != 0]  # Remove the in-between recordings
        X, y = transform_data(data)
        if i == 1: #First training sample
            X_train = X
            y_train = y
        elif i == 9: #First test sample
            X_test = X
            y_test = y
        elif i == 10: #Second test sample
            X_test = np.concatenate((X_test, X), axis=0)
            y_test = np.concatenate((y_test, y), axis=0)
        else: #All other training samples
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

dotimeseriesclassification(X_train, X_test, y_train, y_test)
