import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from pyts.classification import TimeSeriesForest
from pyts.multivariate.classification import MultivariateClassifier
import time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from darts.dataprocessing import dtw
from darts.timeseries import TimeSeries
from scipy.signal import resample

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
        for subject_index, subject_number in enumerate(class_subset['subject_number'].unique()): #Must separate each subject
            subset = class_subset[class_subset['subject_number'] == subject_number]
            subject_X = None
            for j, measurement in enumerate(data.columns.unique()): #Loop for each feature
                if measurement != 'class' and measurement != 'subject_number':
                    class_subset_window = slidingWindow(subset, sampling_rate, measurement)
                    if j == 0:
                        subject_X = class_subset_window
                    elif j == 1:
                        subject_X = np.stack((subject_X, class_subset_window), axis=1)
                    else:
                        subject_X = np.concatenate((subject_X, class_subset_window[:, np.newaxis, :]), axis=1)
            if subject_index == 0:
                X = subject_X
            else:
                X = np.concatenate((X, subject_X), axis=0)
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
    time1 = time.time()
    clf = MultivariateClassifier(TimeSeriesForest(random_state=42, class_weight='balanced', n_jobs=-1))
    clf.fit(X_train, y_train)
    time2 = time.time()
    y_pred = clf.predict(X_test)
    time3 = time.time()
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print(f"Training Time: {time2-time1}")
    print(f"Inference Time: {time3-time2}")

def pad_arrays(arrays):
    max_length = max(arr.shape[0] for arr in arrays)

    padded_arrays = []
    for arr in arrays:
        if arr.shape[0] < max_length:
            pad_width = ((0, max_length - arr.shape[0]),) + ((0, 0),) * (arr.ndim - 1)
            padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
            padded_arrays.append(padded_arr)
        else:
            padded_arrays.append(arr)
    return padded_arrays
def average_series(df):
    X_list = []
    y_list = []
    columns = df.columns.tolist()
    columns.remove('class')
    columns.remove('subject_number')
    for subject_number in df['subject_number'].unique():
        subset = df[df['subject_number'] == subject_number]
        y_list.append(subset['class'].values)
        subset = subset.drop(['class', 'subject_number'], axis=1)
        X_list.append(subset.values)
    X = None
    y = None
    for class_num in classes.keys():
        if class_num != 0:
            X_class = []
            for i, series_y in enumerate(y_list):
                indices = np.where(series_y == class_num)[0]
                X_temp = X_list[i][indices]
                X_class.append(X_temp)
            same_length = all(arr.shape[0] == X_class[0].shape[0] for arr in X_class)
            if not same_length:
                length_adjusted = pad_arrays(X_class)
                average_X_class = np.nanmean(np.stack(length_adjusted, axis=-1), axis=-1)
            else:
                average_X_class = np.nanmean(np.stack(X_class, axis=-1), axis=-1)
            average_y_class = np.full(average_X_class.shape[0], class_num)

            if class_num == 1:
                X = average_X_class
                y = average_y_class
            else:
                X = np.concatenate((X, average_X_class), axis=0)
                y = np.concatenate((y, average_y_class), axis=0)
    new_df = pd.DataFrame(X, columns=columns)
    new_df['class'] = y
    new_df['subject_number'] = 1
    return new_df

def runDTW(data):
    time1 = time.time()
    new_df = pd.DataFrame()
    for class_val in data['class'].unique():
        subset = data[data['class'] == class_val]
        base = subset[subset['subject_number'] == 1]
        base = base.drop(['class', 'subject_number'], axis=1)
        new_class_subset = pd.DataFrame()
        for column in base.columns:
            base_series = TimeSeries.from_values(base[column].values)
            new_series = []
            for subject_number in subset['subject_number'].unique():
                if subject_number != 1:
                    current_series = TimeSeries.from_values(subset[subset['subject_number']==subject_number][column].values)
                    alignment = dtw.dtw(base_series, current_series, multi_grid_radius=10)
                    series_1, series_2 = alignment.warped()
                    #downsampled_1 = resample(series_1.values().flatten(), len(base[column].values))
                    downsampled_2 = resample(series_2.values().flatten(), len(base[column].values))
                    #base_series = TimeSeries.from_values(downsampled_1) # Try without this
                    new_series.append(downsampled_2)
            new_series.insert(0, base_series.values().flatten())
            average_series = np.nanmean(np.stack(new_series, axis=0), axis=0)
            print(column)
            print(average_series.shape)
            new_class_subset[column] = average_series
        new_class_subset['class'] = class_val
        new_df = pd.concat([new_df, new_class_subset])
    new_df['subject_number'] = 1
    print(time.time()-time1)
    return new_df


    # exit()
    # subset_0 = data[data['subject_number'] == 1]
    # base_series = []
    # for class_val in subset_0['class'].unique():
    #     class_subset = subset_0[subset_0['class'] == class_val]
    #     class_subset = class_subset.drop(['class', 'subject_number'], axis=1)
    #     for column in class_subset.columns:
    #         base_series.append(TimeSeries.from_values(class_subset[column].values))
    #
    # new_df = pd.DataFrame()
    # for subject_number in data['subject_number'].unique():
    #     if subject_number != 1:
    #         subset = data[data['subject_number'] == subject_number]
    #         for class_val in data['class'].unique():
    #             class_df = pd.DataFrame()
    #             class_subset = subset[subset['class'] == class_val]
    #             class_subset = class_subset.drop(['class', 'subject_number'], axis=1)
    #             for i, column in enumerate(class_subset.columns):
    #                 series = TimeSeries.from_values(class_subset[column].values)
    #                 alignment = dtw.dtw(base_series[i], series, multi_grid_radius=10)
    #                 series_1, series_2 = alignment.warped()
    #                 downsampled_1 = resample(series_1.values().flatten(), len(class_subset[column].values))
    #                 downsampled_2 = resample(series_2.values().flatten(), len(class_subset[column].values))
    #                 base_series[i] = TimeSeries.from_values(downsampled_1) #Try without this too
    #                 class_df[column] = downsampled_2
    #             class_df['class'] = class_val
    #             class_df['subject_number'] = subject_number
    #             print(class_df.shape)
    #             new_df = pd.concat([new_df, class_df], axis=0)
    #         print("OUT")
    #         print(new_df.shape)
    # exit()
    #
    #
    #
    #
    # print(data[0].shape)
    # series1 = TimeSeries.from_values(data[0])
    # new_series = []
    # for i, series in enumerate(data):
    #     if i > 0:
    #         series2 = TimeSeries.from_values(series)
    #         alignment = dtw.dtw(series1, series2, multi_grid_radius=10)
    #         series1, series2_new = alignment.warped()
    #         new_series.append(series2_new)
    # new_series.insert(0, series1)
    # return new_series
    # subset = data[data['class'] == 8]
    # data1 = subset['chest_acceleration_z'].iloc[:500].values
    # data2 = subset['chest_acceleration_z'].iloc[700:1200].values
    # data1_ts = TimeSeries.from_values(data1)
    # data2_ts = TimeSeries.from_values(data2)
    # # plt.plot(list(range(1,501)), data1)
    # # plt.plot(list(range(1, 501)), data2)
    # # plt.show()
    # alignment = dtw.dtw(data1_ts, data2_ts, multi_grid_radius=10)
    # data1_new, data2_new = alignment.warped()
    # # data1_new.plot(color="blue", label="Series 1")
    # # data2_new.plot(color="orange", label="Series 2")
    # # plt.show()


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
doAveraging = False
doSmoothing = False
doDTW = False
average_contents_x = []
average_contents_y = []
training_data = pd.DataFrame()
testing_data = pd.DataFrame()
for i in range(1,11):
    file_contents = []
    with open('MHEALTHDATASET/mHealth_subject'+str(i)+'.log', 'r') as file: #Read in log file
        print(i)
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            file_contents.append(row)
        data = pd.DataFrame(file_contents, columns=columns)
        data = data.astype(float)
        data['class'] = data['class'].astype(int)

        data = data[data['class'] != 0]  # Remove the in-between recordings
        data['subject_number'] = i
        if i < 9:
            training_data = pd.concat([training_data, data], axis=0)
        else:
            testing_data = pd.concat([testing_data, data], axis=0)
if doDTW:
    training_data = runDTW(training_data)
elif doAveraging:
    training_data = average_series(training_data)
X_train, y_train = transform_data(training_data)
X_test, y_test = transform_data(testing_data)

# if doDTW:
#     runDTW(training_data)
#
#
#
#         if doSmoothing and i < 9:
#             new_data = pd.DataFrame()
#             for class_val in data['class'].unique():
#                 subset = data[data['class'] == class_val]
#                 smaller_df = pd.DataFrame()
#                 for column in data.columns:
#                     if column != 'class':
#                     # data[column] = gaussian_filter(data[column], 1)
#                         window_size = 3
#                         smaller_df[column] = np.convolve(subset[column], np.ones(window_size)/window_size, mode='valid')
#                 smaller_df['class'] = class_val
#                 new_data = pd.concat([new_data, smaller_df], axis=0)
#             data = new_data
#
#         X, y = transform_data(data)
#
#         if doAveraging:
#             if i != 9 and i != 10:
#                 average_contents_y.append(y)
#                 average_contents_x.append(X)
#             else:
#                 if i == 9:  # First test sample
#                     X_test = X
#                     y_test = y
#                 else:  # Second test sample
#                     X_test = np.concatenate((X_test, X), axis=0)
#                     y_test = np.concatenate((y_test, y), axis=0)
#         else:
#             if i == 1: #First training sample
#                 X_train = X
#                 y_train = y
#             elif i == 9: #First test sample
#                 X_test = X
#                 y_test = y
#             elif i == 10: #Second test sample
#                 X_test = np.concatenate((X_test, X), axis=0)
#                 y_test = np.concatenate((y_test, y), axis=0)
#             else: #All other training samples
#                 X_train = np.concatenate((X_train, X), axis=0)
#                 y_train = np.concatenate((y_train, y), axis=0)

dotimeseriesclassification(X_train, X_test, y_train, y_test)
