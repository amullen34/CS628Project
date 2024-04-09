import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from pyts.classification import TimeSeriesForest
from pyts.multivariate.classification import MultivariateClassifier
import matplotlib.pyplot as plt

def slidingWindow(df, sampling_rate, measurement):
    window_size = 5 * sampling_rate
    step_size = sampling_rate
    num_windows = int(len(df) / step_size)
    indexer = np.arange(window_size)[None, :] + step_size * np.arange(num_windows)[:, None]
    num_na = int(window_size / step_size - 1)
    indexer = indexer[:-num_na]
    values = df[measurement].values
    return values[indexer]

def transform_data(data):
    sampling_rate = 50
    X_overall, y_overall = None, None
    for classification in data['class'].unique():
        class_subset = data[data['class'] == classification]
        X = []
        for measurement in [col for col in data.columns if col != 'class']:
            windowed_data = slidingWindow(class_subset, sampling_rate, measurement)
            X.append(windowed_data[..., np.newaxis])
        X = np.concatenate(X, axis=-1)
        y = np.full((X.shape[0],), classification)
        if X_overall is None:
            X_overall, y_overall = X, y
        else:
            X_overall = np.concatenate((X_overall, X), axis=0)
            y_overall = np.concatenate((y_overall, y), axis=0)
    return X_overall, y_overall

def load_and_transform_data(subject_id, columns):
    file_path = f'/content/drive/MyDrive/Colab Notebooks/MHEALTHDATASET/mHealth_subject{subject_id}.log'
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        data = pd.DataFrame(reader, columns=columns)
        data = data.astype(float)
        data['class'] = data['class'].astype(int)
    data = data[data['class'] != 0]
    return transform_data(data)

def evaluate_model(X_train, y_train, X_test, y_test):
    clf = MultivariateClassifier(TimeSeriesForest(random_state=42, class_weight='balanced', n_jobs=-1))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, f1

#columns based on the dataset specification
columns = ['chest_acceleration_x', 'chest_acceleration_y', 'chest_acceleration_z', 'ecg_1', 'ecg_2',
           'left_ankle_acceleration_x', 'left_ankle_acceleration_y', 'left_ankle_acceleration_z',
           'left_ankle_gyro_x', 'left_ankle_gyro_y', 'left_ankle_gyro_z', 'left_ankle_magnetometer_x',
           'left_ankle_magnetometer_y', 'left_ankle_magnetometer_z', 'right_arm_acceleration_x',
           'right_arm_acceleration_y', 'right_arm_acceleration_z', 'right_arm_gyro_x', 'right_arm_gyro_y',
           'right_arm_gyro_z', 'right_arm_magnetometer_x', 'right_arm_magnetometer_y', 'right_arm_magnetometer_z',
           'class']

# Load and transform data for each subject
subject_ids = range(1, 11)  #subjects 1 through 10
accuracies, f1_scores = [], []

for subject_id in subject_ids:
    X, y = load_and_transform_data(subject_id, columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    accuracy, f1 = evaluate_model(X_train, y_train, X_test, y_test)
    accuracies.append(accuracy)
    f1_scores.append(f1)

# Plotting
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(subject_ids, accuracies, marker='o', linestyle='-', color='blue')
plt.title('Accuracy per Subject')
plt.xlabel('Subject ID')
plt.ylabel('Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(subject_ids, f1_scores, marker='o', linestyle='-', color='red')
plt.title('F1 Score per Subject')
plt.xlabel('Subject ID')
plt.ylabel('F1 Score')
plt.grid(True)

plt.tight_layout()
plt.show()
