import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from pyts.classification import TimeSeriesForest


def slidingWindow(df, sampling_rate, measurement):
    window_size = 5 * sampling_rate #Window size 5 seconds
    step_size = sampling_rate #Step size 1 second
    # num_windows = int((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / (step_size/sampling_rate))
    num_windows = int(len(df) / step_size)
    indexer = np.arange(window_size)[None, :] + step_size * np.arange(num_windows)[:, None] #Create indices for sliding window
    num_na = int(window_size/step_size - 1) #Remove the last few rows that will have missing values
    indexer = indexer[:-num_na]

    values = df[measurement].values
    df_new = pd.DataFrame(values[indexer], columns=[f'Reading_{i}' for i in range(window_size)]) #Convert to dataframe

    df_new['class'] = df.iloc[0]['class']
    return df_new

def transform_data(data):
    sampling_rate = 50
    measurement = 'chest_acceleration_z'
    df = data[[measurement, 'class']]
    new_df = pd.DataFrame()
    for classification in df['class'].unique():
        class_subset = df[df['class'] == classification]
        class_subset_window = slidingWindow(class_subset, sampling_rate, measurement)
        new_df = pd.concat([new_df, class_subset_window])
    new_df = (new_df.reset_index()).drop(['index'], axis=1)
    return new_df

def dotimeseriesclassification(train, test):

    X_train = train.drop(['class'], axis=1).values
    y_train = train['class'].values
    X_test = test.drop(['class'], axis=1).values
    y_test = test['class'].values
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    clf = TimeSeriesForest(random_state=42, class_weight='balanced', n_jobs=-1)
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

training_data = pd.DataFrame()
test_data = pd.DataFrame()
for i in range(1,11):
    file_contents = []
    with open('MHEALTHDATASET/mHealth_subject'+str(i)+'.log', 'r') as file: #Read in log file
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            file_contents.append(row)
        data = pd.DataFrame(file_contents, columns=columns)
        data = data.astype(float)
        data['class'] = data['class'].astype(int)
        # arbitrary_start_time = pd.Timestamp('2024-01-01 00:00:00')
        # data['timestamp'] = pd.date_range(start=arbitrary_start_time, periods=len(data),
        #                                   freq='20ms')  # 50 Hz = 20 milliseconds

        data = data[data['class'] != 0]  # Remove the in-between recordings
        transformed_data = transform_data(data)
        print(len(transformed_data))
        if i == 9 or i == 10:
            test_data = pd.concat([test_data, transformed_data])
        else:
            training_data = pd.concat([training_data, transformed_data])


dotimeseriesclassification(training_data, test_data)
#When random time periods are sampled from all data, performance is very good
#When certain people are separated and designated as test, it doesn't work as well
#It doesn't generalize well to other people, how can we fix that?
