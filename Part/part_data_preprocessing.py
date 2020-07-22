import pandas as pd
import numpy as np
import math
import sys
from scipy.stats import pearsonr


def data_check(data):

    data_checked = []
    for i in range(len(data)):
        NaN_count = 0
        data_hour = data[i]
        for j in range(len(data_hour)):
            NaN_judge = math.isnan(data_hour[j])
            if NaN_judge is True:
                NaN_count +=1
        if NaN_count == 0:
            data_checked.append(data_hour)
    print(len(data_checked))

    return data_checked


def data_check_test(data):

    # data_back = np.array(data).T.tolist()
    data_checked = np.array(data).T.tolist()

    for i in range(len(data_checked)):
        data_feature = data_checked[i]
        for j in range(len(data_feature)):
            NaN_judge = math.isnan(data_feature[j])
            if NaN_judge is True or data_feature[j] == 0:
                if j == 0:  # mean of first.
                    h1 = 1
                    while math.isnan(data_checked[i][j + h1]) is True:
                        h1 = h1 + 1
                    mean = data_checked[i][j + h1]
                elif j == len(data) - 1:  # mean of last.
                    h2 = - 1
                    while math.isnan(data_checked[i][j + h2]) is True:
                        h2 = h2 - 1
                    mean = data_checked[i][j + h2]
                else:
                    h1 = 1
                    while math.isnan(data_checked[i][j + h1]) is True:
                        h1 = h1 + 1
                    h2 = - 1
                    while math.isnan(data_checked[i][j + h2]) is True:
                        h2 = h2 - 1
                    mean = (data_checked[i][j + h1] + data_checked[i][j + h2]) / 2
                data_feature[j] = mean
        data_checked[i] = data_feature

    data_checked = np.array(data_checked).T.tolist()

    return data_checked


def data_interval(data, itv):

    data_itved = []
    for i in range(len(data)):
        if (i % itv) == 0:
            data_itved.append(data[i])

    return data_itved


def read_csv(filename_list, trainNum, testNum, startNum, file_num, interval):

    targetData = None
    SO2 = None
    NO2 = None
    CO = None
    O3 = None
    TEMP = None
    PRES = None
    DEWP = None
    RAIN = None
    WSPM = None

    for i in range(file_num):
        filename = filename_list[0] + str(i + 1) + filename_list[1]
        dataset = pd.read_csv(filename, encoding='gbk')

        # 输入数据
        file_PM2 = dataset.iloc[:, 5]  # PM2.5 2k
        file_PM10 = dataset.iloc[:, 6]   # PM10
        file_SO2 = dataset.iloc[:, 7]   # SO2 2k
        file_NO2 = dataset.iloc[:, 8]
        file_CO = dataset.iloc[:, 9]
        file_O3 = dataset.iloc[:, 10]
        file_TEMP = dataset.iloc[:, 11]
        file_PRES = dataset.iloc[:, 12]
        file_DEWP = dataset.iloc[:, 13]
        file_RAIN = dataset.iloc[:, 14]
        file_WSPM = dataset.iloc[:, 16]

        # 预测对象
        file_targetData = file_PM2

        file_targetData = np.array(file_targetData).reshape(-1, 1)
        file_SO2 = np.array(file_SO2).reshape(-1, 1)
        file_NO2 = np.array(file_NO2).reshape(-1, 1)
        file_CO = np.array(file_CO).reshape(-1, 1)
        file_O3 = np.array(file_O3).reshape(-1, 1)
        file_TEMP = np.array(file_TEMP).reshape(-1, 1)
        file_PRES = np.array(file_PRES).reshape(-1, 1)
        file_DEWP = np.array(file_DEWP).reshape(-1, 1)
        file_RAIN = np.array(file_RAIN).reshape(-1, 1)
        file_WSPM = np.array(file_WSPM).reshape(-1, 1)

        if i == 0:
            targetData = file_targetData
            SO2 = file_SO2
            NO2 = file_NO2
            CO = file_CO
            O3 = file_O3
            TEMP = file_TEMP
            PRES = file_PRES
            DEWP = file_DEWP
            RAIN = file_RAIN
            WSPM = file_WSPM
        else:
            targetData = np.r_[targetData, file_targetData]
            SO2 = np.r_[SO2, file_SO2]
            NO2 = np.r_[NO2, file_NO2]
            CO = np.r_[CO, file_CO]
            O3 = np.r_[O3, file_O3]
            TEMP = np.r_[TEMP, file_TEMP]
            PRES = np.r_[PRES, file_PRES]
            DEWP = np.r_[DEWP, file_DEWP]
            RAIN = np.r_[RAIN, file_RAIN]
            WSPM = np.r_[WSPM, file_WSPM]

    allX = np.c_[SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM]
    all_data = np.c_[allX, targetData]
    all_data = all_data.tolist()
    # all_data_checked = data_check(all_data)
    all_data_checked = data_check_test(all_data)
    all_data_checked = data_interval(all_data_checked, interval)
    all_data_checked = np.array(all_data_checked)

    print("Total Data:", all_data_checked.shape)
    data_shape = all_data_checked.shape
    max_dataset = data_shape[0]
    if max_dataset < (startNum + testNum + trainNum):
        print('This Dataset is not enough for such trainNum & testNum.')
        sys.exit(1)

    return all_data_checked


def read_csv_all(filename_list, trainNum, testNum, startNum, file_num, interval):

    PM2 = None
    PM10 = None
    SO2 = None
    NO2 = None
    CO = None
    O3 = None
    TEMP = None
    PRES = None
    DEWP = None

    for i in range(file_num):
        filename = filename_list[0] + str(i + 1) + filename_list[1]
        dataset = pd.read_csv(filename, encoding='gbk')

        # 输入数据
        file_PM2 = np.array(dataset.iloc[:, 5]).reshape(-1, 1)
        file_PM10 = np.array(dataset.iloc[:, 6]).reshape(-1, 1)
        file_SO2 = np.array(dataset.iloc[:, 7]).reshape(-1, 1)
        file_NO2 = np.array(dataset.iloc[:, 8]).reshape(-1, 1)
        file_CO = np.array(dataset.iloc[:, 9]).reshape(-1, 1)
        file_O3 = np.array(dataset.iloc[:, 10]).reshape(-1, 1)
        file_TEMP = np.array(dataset.iloc[:, 11]).reshape(-1, 1)
        file_PRES = np.array(dataset.iloc[:, 12]).reshape(-1, 1)
        file_DEWP = np.array(dataset.iloc[:, 13]).reshape(-1, 1)

        if i == 0:
            PM2 = file_PM2
            PM10 = file_PM10
            SO2 = file_SO2
            NO2 = file_NO2
            CO = file_CO
            O3 = file_O3
            TEMP = file_TEMP
            PRES = file_PRES
            DEWP = file_DEWP
        else:
            PM2 = np.r_[PM2, file_PM2]
            PM10 = np.r_[PM10, file_PM10]
            SO2 = np.r_[SO2, file_SO2]
            NO2 = np.r_[NO2, file_NO2]
            CO = np.r_[CO, file_CO]
            O3 = np.r_[O3, file_O3]
            TEMP = np.r_[TEMP, file_TEMP]
            PRES = np.r_[PRES, file_PRES]
            DEWP = np.r_[DEWP, file_DEWP]

    all_data = [PM2, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP]
    all_data_name_list = ['PM2', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP']
    all_data_checked_list = []

    for b in range(len(all_data)):

        feature_data = all_data[b].tolist()
        all_data_checked = data_check_test(feature_data)
        all_data_checked = data_interval(all_data_checked, interval)
        all_data_checked = np.array(all_data_checked)
        all_data_checked_list.append(all_data_checked.reshape(-1, ))

        print(all_data_name_list[b], all_data_checked.shape)
        data_shape = all_data_checked.shape
        max_dataset = data_shape[0]
        if max_dataset < (startNum + testNum + trainNum):
            print('This Dataset is not enough for such trainNum & testNum.')
            print('Insufficient data: ', all_data_name_list[b])
            sys.exit(1)

    return all_data_checked_list


def read_csv_PM2(filename_list, trainNum, testNum, startNum, file_num, interval):

    targetData = None

    for i in range(file_num):
        filename = filename_list[0] + str(i + 1) + filename_list[1]
        dataset = pd.read_csv(filename, encoding='gbk')

        # 输入数据
        file_PM2 = dataset.iloc[:, 5]  # PM2.5 2k
        file_PM10 = dataset.iloc[:, 6]   # PM10
        file_SO2 = dataset.iloc[:, 7]   # SO2 2k
        file_NO2 = dataset.iloc[:, 8]
        file_CO = dataset.iloc[:, 9]
        file_O3 = dataset.iloc[:, 10]
        file_TEMP = dataset.iloc[:, 11]
        file_PRES = dataset.iloc[:, 12]
        file_DEWP = dataset.iloc[:, 13]

        # 预测对象
        file_targetData = file_PM2
        file_targetData = np.array(file_targetData).reshape(-1, 1)

        if i == 0:
            targetData = file_targetData
        else:
            targetData = np.r_[targetData, file_targetData]

    all_data = targetData
    all_data = all_data.tolist()
    # all_data_checked = data_check(all_data)
    all_data_checked = data_check_test(all_data)
    all_data_checked = data_interval(all_data_checked, interval)
    all_data_checked = np.array(all_data_checked)

    print("Total Data:", all_data_checked.shape)
    data_shape = all_data_checked.shape
    max_dataset = data_shape[0]
    if max_dataset < (startNum + testNum + trainNum):
        print('This Dataset is not enough for such trainNum & testNum.')
        sys.exit(1)

    return all_data_checked.reshape(-1, )


def read_csv_PM10(filename_list, trainNum, testNum, startNum, file_num, interval):

    targetData = None

    for i in range(file_num):
        filename = filename_list[0] + str(i + 1) + filename_list[1]
        dataset = pd.read_csv(filename, encoding='gbk')

        # 输入数据
        file_PM2 = dataset.iloc[:, 5]  # PM2.5 2k
        file_PM10 = dataset.iloc[:, 6]   # PM10
        file_SO2 = dataset.iloc[:, 7]   # SO2 2k
        file_NO2 = dataset.iloc[:, 8]
        file_CO = dataset.iloc[:, 9]
        file_O3 = dataset.iloc[:, 10]
        file_TEMP = dataset.iloc[:, 11]
        file_PRES = dataset.iloc[:, 12]
        file_DEWP = dataset.iloc[:, 13]

        # 预测对象
        file_targetData = file_PM10
        file_targetData = np.array(file_targetData).reshape(-1, 1)

        if i == 0:
            targetData = file_targetData
        else:
            targetData = np.r_[targetData, file_targetData]

    all_data = targetData
    all_data = all_data.tolist()
    # all_data_checked = data_check(all_data)
    all_data_checked = data_check_test(all_data)
    all_data_checked = data_interval(all_data_checked, interval)
    all_data_checked = np.array(all_data_checked)

    print("Total Data:", all_data_checked.shape)
    data_shape = all_data_checked.shape
    max_dataset = data_shape[0]
    if max_dataset < (startNum + testNum + trainNum):
        print('This Dataset is not enough for such trainNum & testNum.')
        sys.exit(1)

    return all_data_checked.reshape(-1, )


def read_csv_SO2(filename_list, trainNum, testNum, startNum, file_num, interval):

    targetData = None

    for i in range(file_num):
        filename = filename_list[0] + str(i + 1) + filename_list[1]
        dataset = pd.read_csv(filename, encoding='gbk')

        # 输入数据
        file_PM2 = dataset.iloc[:, 5]  # PM2.5 2k
        file_PM10 = dataset.iloc[:, 6]   # PM10
        file_SO2 = dataset.iloc[:, 7]   # SO2 2k
        file_NO2 = dataset.iloc[:, 8]
        file_CO = dataset.iloc[:, 9]
        file_O3 = dataset.iloc[:, 10]
        file_TEMP = dataset.iloc[:, 11]
        file_PRES = dataset.iloc[:, 12]
        file_DEWP = dataset.iloc[:, 13]

        # 预测对象
        file_targetData = file_SO2
        file_targetData = np.array(file_targetData).reshape(-1, 1)

        if i == 0:
            targetData = file_targetData
        else:
            targetData = np.r_[targetData, file_targetData]

    all_data = targetData
    all_data = all_data.tolist()
    # all_data_checked = data_check(all_data)
    all_data_checked = data_check_test(all_data)
    all_data_checked = data_interval(all_data_checked, interval)
    all_data_checked = np.array(all_data_checked)

    print("Total Data:", all_data_checked.shape)
    data_shape = all_data_checked.shape
    max_dataset = data_shape[0]
    if max_dataset < (startNum + testNum + trainNum):
        print('This Dataset is not enough for such trainNum & testNum.')
        sys.exit(1)

    return all_data_checked.reshape(-1, )


def read_csv_NO2(filename_list, trainNum, testNum, startNum, file_num, interval):

    targetData = None

    for i in range(file_num):
        filename = filename_list[0] + str(i + 1) + filename_list[1]
        dataset = pd.read_csv(filename, encoding='gbk')

        # 输入数据
        file_PM2 = dataset.iloc[:, 5]  # PM2.5 2k
        file_PM10 = dataset.iloc[:, 6]   # PM10
        file_SO2 = dataset.iloc[:, 7]   # SO2 2k
        file_NO2 = dataset.iloc[:, 8]
        file_CO = dataset.iloc[:, 9]
        file_O3 = dataset.iloc[:, 10]
        file_TEMP = dataset.iloc[:, 11]
        file_PRES = dataset.iloc[:, 12]
        file_DEWP = dataset.iloc[:, 13]

        # 预测对象
        file_targetData = file_NO2
        file_targetData = np.array(file_targetData).reshape(-1, 1)

        if i == 0:
            targetData = file_targetData
        else:
            targetData = np.r_[targetData, file_targetData]

    all_data = targetData
    all_data = all_data.tolist()
    # all_data_checked = data_check(all_data)
    all_data_checked = data_check_test(all_data)
    all_data_checked = data_interval(all_data_checked, interval)
    all_data_checked = np.array(all_data_checked)

    print("Total Data:", all_data_checked.shape)
    data_shape = all_data_checked.shape
    max_dataset = data_shape[0]
    if max_dataset < (startNum + testNum + trainNum):
        print('This Dataset is not enough for such trainNum & testNum.')
        sys.exit(1)

    return all_data_checked.reshape(-1, )


def read_csv_CO(filename_list, trainNum, testNum, startNum, file_num, interval):

    targetData = None

    for i in range(file_num):
        filename = filename_list[0] + str(i + 1) + filename_list[1]
        dataset = pd.read_csv(filename, encoding='gbk')

        # 输入数据
        file_PM2 = dataset.iloc[:, 5]  # PM2.5 2k
        file_PM10 = dataset.iloc[:, 6]   # PM10
        file_SO2 = dataset.iloc[:, 7]   # SO2 2k
        file_NO2 = dataset.iloc[:, 8]
        file_CO = dataset.iloc[:, 9]
        file_O3 = dataset.iloc[:, 10]
        file_TEMP = dataset.iloc[:, 11]
        file_PRES = dataset.iloc[:, 12]
        file_DEWP = dataset.iloc[:, 13]

        # 预测对象
        file_targetData = file_CO
        file_targetData = np.array(file_targetData).reshape(-1, 1)

        if i == 0:
            targetData = file_targetData
        else:
            targetData = np.r_[targetData, file_targetData]

    all_data = targetData
    all_data = all_data.tolist()
    # all_data_checked = data_check(all_data)
    all_data_checked = data_check_test(all_data)
    all_data_checked = data_interval(all_data_checked, interval)
    all_data_checked = np.array(all_data_checked)

    print("Total Data:", all_data_checked.shape)
    data_shape = all_data_checked.shape
    max_dataset = data_shape[0]
    if max_dataset < (startNum + testNum + trainNum):
        print('This Dataset is not enough for such trainNum & testNum.')
        sys.exit(1)

    return all_data_checked.reshape(-1, )


def read_csv_O3(filename_list, trainNum, testNum, startNum, file_num, interval):

    targetData = None

    for i in range(file_num):
        filename = filename_list[0] + str(i + 1) + filename_list[1]
        dataset = pd.read_csv(filename, encoding='gbk')

        # 输入数据
        file_PM2 = dataset.iloc[:, 5]  # PM2.5 2k
        file_PM10 = dataset.iloc[:, 6]   # PM10
        file_SO2 = dataset.iloc[:, 7]   # SO2 2k
        file_NO2 = dataset.iloc[:, 8]
        file_CO = dataset.iloc[:, 9]
        file_O3 = dataset.iloc[:, 10]
        file_TEMP = dataset.iloc[:, 11]
        file_PRES = dataset.iloc[:, 12]
        file_DEWP = dataset.iloc[:, 13]

        # 预测对象
        file_targetData = file_O3
        file_targetData = np.array(file_targetData).reshape(-1, 1)

        if i == 0:
            targetData = file_targetData
        else:
            targetData = np.r_[targetData, file_targetData]

    all_data = targetData
    all_data = all_data.tolist()
    # all_data_checked = data_check(all_data)
    all_data_checked = data_check_test(all_data)
    all_data_checked = data_interval(all_data_checked, interval)
    all_data_checked = np.array(all_data_checked)

    print("Total Data:", all_data_checked.shape)
    data_shape = all_data_checked.shape
    max_dataset = data_shape[0]
    if max_dataset < (startNum + testNum + trainNum):
        print('This Dataset is not enough for such trainNum & testNum.')
        sys.exit(1)

    return all_data_checked.reshape(-1, )


def read_csv_TEMP(filename_list, trainNum, testNum, startNum, file_num, interval):

    targetData = None

    for i in range(file_num):
        filename = filename_list[0] + str(i + 1) + filename_list[1]
        dataset = pd.read_csv(filename, encoding='gbk')

        # 输入数据
        file_PM2 = dataset.iloc[:, 5]  # PM2.5 2k
        file_PM10 = dataset.iloc[:, 6]   # PM10
        file_SO2 = dataset.iloc[:, 7]   # SO2 2k
        file_NO2 = dataset.iloc[:, 8]
        file_CO = dataset.iloc[:, 9]
        file_O3 = dataset.iloc[:, 10]
        file_TEMP = dataset.iloc[:, 11]
        file_PRES = dataset.iloc[:, 12]
        file_DEWP = dataset.iloc[:, 13]

        # 预测对象
        file_targetData = file_TEMP
        file_targetData = np.array(file_targetData).reshape(-1, 1)

        if i == 0:
            targetData = file_targetData
        else:
            targetData = np.r_[targetData, file_targetData]

    all_data = targetData
    all_data = all_data.tolist()
    # all_data_checked = data_check(all_data)
    all_data_checked = data_check_test(all_data)
    all_data_checked = data_interval(all_data_checked, interval)
    all_data_checked = np.array(all_data_checked)

    print("Total Data:", all_data_checked.shape)
    data_shape = all_data_checked.shape
    max_dataset = data_shape[0]
    if max_dataset < (startNum + testNum + trainNum):
        print('This Dataset is not enough for such trainNum & testNum.')
        sys.exit(1)

    return all_data_checked.reshape(-1, )


def read_csv_PRES(filename_list, trainNum, testNum, startNum, file_num, interval):

    targetData = None

    for i in range(file_num):
        filename = filename_list[0] + str(i + 1) + filename_list[1]
        dataset = pd.read_csv(filename, encoding='gbk')

        # 输入数据
        file_PM2 = dataset.iloc[:, 5]  # PM2.5 2k
        file_PM10 = dataset.iloc[:, 6]   # PM10
        file_SO2 = dataset.iloc[:, 7]   # SO2 2k
        file_NO2 = dataset.iloc[:, 8]
        file_CO = dataset.iloc[:, 9]
        file_O3 = dataset.iloc[:, 10]
        file_TEMP = dataset.iloc[:, 11]
        file_PRES = dataset.iloc[:, 12]
        file_DEWP = dataset.iloc[:, 13]

        # 预测对象
        file_targetData = file_PRES
        file_targetData = np.array(file_targetData).reshape(-1, 1)

        if i == 0:
            targetData = file_targetData
        else:
            targetData = np.r_[targetData, file_targetData]

    all_data = targetData
    all_data = all_data.tolist()
    # all_data_checked = data_check(all_data)
    all_data_checked = data_check_test(all_data)
    all_data_checked = data_interval(all_data_checked, interval)
    all_data_checked = np.array(all_data_checked)

    print("Total Data:", all_data_checked.shape)
    data_shape = all_data_checked.shape
    max_dataset = data_shape[0]
    if max_dataset < (startNum + testNum + trainNum):
        print('This Dataset is not enough for such trainNum & testNum.')
        sys.exit(1)

    return all_data_checked.reshape(-1, )


def read_csv_DEWP(filename_list, trainNum, testNum, startNum, file_num, interval):

    targetData = None

    for i in range(file_num):
        filename = filename_list[0] + str(i + 1) + filename_list[1]
        dataset = pd.read_csv(filename, encoding='gbk')

        # 输入数据
        file_PM2 = dataset.iloc[:, 5]  # PM2.5 2k
        file_PM10 = dataset.iloc[:, 6]   # PM10
        file_SO2 = dataset.iloc[:, 7]   # SO2 2k
        file_NO2 = dataset.iloc[:, 8]
        file_CO = dataset.iloc[:, 9]
        file_O3 = dataset.iloc[:, 10]
        file_TEMP = dataset.iloc[:, 11]
        file_PRES = dataset.iloc[:, 12]
        file_DEWP = dataset.iloc[:, 13]

        # 预测对象
        file_targetData = file_DEWP
        file_targetData = np.array(file_targetData).reshape(-1, 1)

        if i == 0:
            targetData = file_targetData
        else:
            targetData = np.r_[targetData, file_targetData]

    all_data = targetData
    all_data = all_data.tolist()
    # all_data_checked = data_check(all_data)
    all_data_checked = data_check_test(all_data)
    all_data_checked = data_interval(all_data_checked, interval)
    all_data_checked = np.array(all_data_checked)

    print("Total Data:", all_data_checked.shape)
    data_shape = all_data_checked.shape
    max_dataset = data_shape[0]
    if max_dataset < (startNum + testNum + trainNum):
        print('This Dataset is not enough for such trainNum & testNum.')
        sys.exit(1)

    return all_data_checked.reshape(-1, )


def create_time_series(data, time_step):
    train_num = len(data)
    TS_X = []

    for i in range(train_num - time_step):
        b = data[i:(i + time_step), 0]
        TS_X.append(b)

    TS_X = np.array(TS_X)
    return TS_X


def create_time_series_3d(data, time_step):

    shape_x = data[0].shape[0]
    shape_y = data[0].shape[1]

    time_series_list = []
    for i in range(len(data)):
        add_list = []
        for j in range(time_step):
            add_index = i - time_step + j
            if add_index < 0:
                add_index = 0
            add_list.append(data[add_index])
            # add_list.append(add_index)
        add_list = np.array(add_list)
        add_list = add_list.reshape(shape_x, shape_y, time_step)
        time_series_list.append(add_list)

    return time_series_list


def create_data(data, train_num, time_step):
    TS_X = []

    for i in range(data.shape[0] - time_step):
        b = data[i:(i + time_step), 0]
        TS_X.append(b)

    dataX1 = TS_X[:train_num]
    dataX2 = TS_X[train_num:]
    dataY1 = data[time_step: train_num + time_step, 0]
    dataY2 = data[train_num + time_step:, 0]

    return np.array(dataX1), np.array(dataY1), np.array(dataX2), np.array(dataY2)


def create_data_neo(data, train_num, ahead_num):

    TS_X = []

    for i in range(data.shape[0] - ahead_num):
        b = data[i:(i + ahead_num), 0]
        TS_X.append(b)

    dataX1 = TS_X[:train_num]
    dataX2 = TS_X[train_num:]
    dataY1 = data[ahead_num:ahead_num + train_num, 0]
    dataY2 = data[ahead_num + train_num:, 0]

    return np.array(dataX1), np.array(dataY1), np.array(dataX2), np.array(dataY2)


def create_data_att(data, train_num):

    dataX1 = np.array(data[:train_num, 0]).T.reshape(1, -1)
    dataX2 = np.array(data[train_num:, 0]).T.reshape(1, -1)
    dataY1 = np.array(data[:train_num, 0]).T.reshape(1, -1)
    dataY2 = np.array(data[train_num:, 0]).T.reshape(1, -1)

    return np.array(dataX1), np.array(dataY1), np.array(dataX2), np.array(dataY2)


def dataprocess_3d(coeffs_list):
    
    list_3s = []
    for i in range(len(coeffs_list)):
        list_2s = []
        for j in range(len(coeffs_list[i][0])):
            list_1s = []
            for h in range(len(coeffs_list[i])):
                list_1s.append(coeffs_list[i][h][j])
            list_2s.append(np.array(list_1s).T)
        list_3s.append(list_2s)

    list_4s = []
    for i in range(len(list_3s[0])):
        data_2d = list_3s[0][i]
        for j in range(len(list_3s) - 1):
            data_2d = np.r_[data_2d, list_3s[j + 1][i]]
        list_4s.append(data_2d)

    return list_4s


def create_datasqr(data, ahead, trainNum, testNum):

    data_sorted = []
    x_processed = []
    for i in range(len(data[0])):
        data_sorted_lv = []
        for j in range(len(data)):
            data_sorted_lv.append(data[j][i])
        data_sorted_lv = np.array(data_sorted_lv).reshape(ahead, -1).T.tolist()
        x_processed_lv = []
        for h in range(len(data_sorted_lv)):
            add_list = []
            for l in range(ahead):
                add_index = h - ahead + l
                if add_index < 0:
                    add_index = 0
                add_list.append(data_sorted_lv[add_index])
            # add_list = np.array(add_list).T.reshape(ahead, ahead, -1)
            x_processed_lv.append(add_list)
            # data_sorted_lv[h] = np.array(data_sorted_lv[h]).reshape(-1, 1)
        data_sorted.append(data_sorted_lv)
        x_processed.append(x_processed_lv)

    data_processed = []
    for i in range(len(data[0])):
        x = x_processed[i]
        y = data[0][i]
        for j in range(len(x)):
            x[j] = np.array(x[j]).T.reshape(ahead, ahead, -1)
        trainX = np.array(x[: trainNum])
        testX = np.array(x[trainNum: trainNum + testNum])
        trainY = y[: trainNum].reshape(-1, )
        data_processed_lv = [trainX, trainY, testX]
        data_processed.append(data_processed_lv)

    return data_processed


def get_fake_predict(predict_true):

    predict_fake = []
    predict_true_list = predict_true.tolist()

    for i in range(len(predict_true_list) - 1):
        predict_fake.append(predict_true_list[i + 1])

    predict_fake.append(predict_true_list[-1])
    predict_fake = np.array(predict_fake)

    return predict_fake


def kpr(array_a, array_b):

    array_a = array_a.reshape(-1, 1).T
    array_a = array_a.tolist()[0]
    array_b = array_b.reshape(-1, 1).T
    array_b = array_b.tolist()[0]

    sim = pearsonr(array_a, array_b)
    # sim = np.corrcoef(array_a, array_b)

    return abs(sim[0])









