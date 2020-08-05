
import csv
import warnings
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor    #### MLP 感知机 ####
from sklearn.tree import ExtraTreeRegressor        #### ExtraTree 极端随机树回归####
from sklearn import tree                           #### 决策树回归 ####
from sklearn.ensemble import BaggingRegressor      #### Bagging回归 ####
from sklearn.ensemble import AdaBoostRegressor     #### Adaboost回归
from sklearn import linear_model                   #### 线性回归####
from sklearn import svm                            #### SVM回归####
from sklearn import ensemble                       #### Adaboost回归####  ####3.7GBRT回归####  ####3.5随机森林回归####
from sklearn import neighbors                      #### KNN回归####

from Model.model_major import *

from pyhht.emd import EMD
from Support.support_wavelet import *

from Part.part_evaluate import *
from Part.part_data_preprocessing import *

warnings.filterwarnings("ignore")


def pre_model(model, trainX, trainY, testX):

    # time_callback = TimeHistory()
    model.fit(trainX, trainY)
    # print(time_callback.totaltime)

    predict = model.predict(testX)
    return predict


########################################################################


def load_data_ts(trainNum, testNum, startNum, data):
    print('General_data loading.')

    global ahead_num
    # all_data_checked = data

    targetData = data

    # 处理预测信息，划分训练集和测试集
    # PM = targetData[startNum : startNum + trainNum + testNum]
    # PM = np.array(PM).reshape(-1, 1)
    targetData = targetData[startNum + 1: startNum + trainNum + testNum + 1]
    targetData = np.array(targetData).reshape(-1, 1)

    # #归一化，每个特征分开归一化
    global scaler_target
    scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)
    targetData = scaler_target.fit_transform(targetData)
    # print("targetData:", targetData.shape)

    # scaler_PM = StandardScaler(copy=True, with_mean=True, with_std=True)
    # PM = scaler_PM.fit_transform(PM)
    # print("PM:", PM.shape)

    global x_mode

    time_series_y = create_time_series(targetData, ahead_num)
    allX = np.c_[time_series_y]
    allX = allX.T

    ###########======================================

    trainX = allX[:, : trainNum]
    trainY = targetData.T[:, ahead_num: trainNum + ahead_num]
    testX = allX[:, trainNum:]
    testY = targetData.T[:, trainNum + ahead_num: (trainNum + testNum)]

    # print("allX:", allX.shape)
    # print("trainX:", trainX.shape)
    # print("trainY:", trainY.shape)
    # print("testX:", testX.shape)
    # print("testY:", testY.shape)

    trainY = trainY.flatten()  # 降维
    testY = testY.flatten()  # 降维
    trainX = trainX.T
    testX = testX.T

    print('load_data complete.\n')

    return trainX, trainY, testX, testY


def load_data_emd(trainNum, testNum, startNum, data):
    print('EMD_data loading.')

    global ahead_num
    # all_data_checked = data

    targetData = data

    # 处理预测信息，划分训练集和测试集
    targetData = targetData[startNum + 1: startNum + trainNum + testNum + 1]
    targetData = np.array(targetData).reshape(-1, 1)

    # #归一化，每个特征分开归一化
    global scaler_target
    scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)
    targetData = scaler_target.fit_transform(targetData)

    decomposer = EMD(targetData)
    imfs = decomposer.decompose()
    # plot_imfs(targetData, imfs)
    data_decomposed = imfs.tolist()

    for h1 in range(len(data_decomposed)):
        data_decomposed[h1] = np.array(data_decomposed[h1]).reshape(-1, 1)
    for h2 in range(len(data_decomposed)):
        trainX, trainY, testX, testY = create_data(data_decomposed[h2], trainNum, ahead_num)
        dataset_imf = [trainX, trainY, testX, testY]
        data_decomposed[h2] = dataset_imf

    print('load_data complete.\n')

    return data_decomposed


def load_data_wvlt(trainNum, testNum, startNum, data):
    print('wavelet_data loading.')

    global ahead_num
    # all_data_checked = data
    targetData = data

    # 处理预测信息，划分训练集和测试集
    targetData = targetData[startNum + 1: startNum + trainNum + testNum + 1]
    targetData = np.array(targetData).reshape(-1, 1)

    # #归一化，每个特征分开归一化
    global scaler_target
    scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)
    targetData = scaler_target.fit_transform(targetData)

    testY = targetData[trainNum: (trainNum + testNum), :]
    wavefun = pywt.Wavelet('db1')
    global wvlt_lv

    coeffs = swt_decom(targetData, wavefun, wvlt_lv)

    ### 测试滤波效果
    wvlt_level_list = []
    for wvlt_level in range(len(coeffs)):
        wvlt_trainX, wvlt_trainY, wvlt_testX, wvlt_testY = create_data(coeffs[wvlt_level], trainNum, ahead_num)
        wvlt_level_part = [wvlt_trainX, wvlt_trainY, wvlt_testX, wvlt_testY]
        wvlt_level_list.append(wvlt_level_part)

    print('load_data complete.\n')

    return wvlt_level_list, testY


#########################################################################


def Decide_Tree(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'decideTree'
    model_name_short = 'dTr'
    print(model_name + ' Start.')

    model_DecisionTreeRegressor = tree.DecisionTreeRegressor()  # 决策树
    predict_decideTree = pre_model(model_DecisionTreeRegressor, x_train, y_train, x_test)
    predict_decideTree = scaler_target.inverse_transform(predict_decideTree)
    flag_decideTree = deal_flag(predict_decideTree, minLen)
    accuracy_decideTree = deal_accuracy(y_rampflag, flag_decideTree)

    # mae_decideTree = MAE1(dataY, predict_decideTree)
    # rmse_decideTree = RMSE1(dataY, predict_decideTree)
    # mape_decideTree = MAPE1(dataY, predict_decideTree)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_decideTree: {}'.format(mae_decideTree)
    # eva_output += '\nrmse_decideTree: {}'.format(rmse_decideTree)
    # eva_output += '\nmape_decideTree: {}'.format(mape_decideTree)
    # eva_output += '\naccuracy_decideTree: {}'.format(accuracy_decideTree)
    #
    # result_all.append(['dTr', mae_decideTree, rmse_decideTree, mape_decideTree, accuracy_decideTree])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name, model_name_short, dataY, predict_decideTree, accuracy_decideTree)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_decideTree


def Random_forest(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'randomForest'
    model_name_short = 'rdF'
    print(model_name + ' Start.')

    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=50)  # 随机森林
    predict_randomForest = pre_model(model_RandomForestRegressor, x_train, y_train, x_test)
    predict_randomForest = scaler_target.inverse_transform(predict_randomForest)
    flag_randomForest = deal_flag(predict_randomForest, minLen)
    accuracy_randomForest = deal_accuracy(y_rampflag, flag_randomForest)

    # rmse_randomForest = RMSE1(dataY, predict_randomForest)
    # mape_randomForest = MAPE1(dataY, predict_randomForest)
    # mae_randomForest = MAE1(dataY, predict_randomForest)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_randomForest: {}'.format(mae_randomForest)
    # eva_output += '\nrmse_randomForest: {}'.format(rmse_randomForest)
    # eva_output += '\nmape_randomForest: {}'.format(mape_randomForest)
    # eva_output += '\naccuracy_randomForest: {}'.format(accuracy_randomForest)
    # result_all.append(['rdF', mae_randomForest, rmse_randomForest, mape_randomForest, accuracy_randomForest])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name, model_name_short, dataY, predict_randomForest,
                                        accuracy_randomForest)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_randomForest


def Linear_Regression(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'linear'
    model_name_short = 'lin'
    print(model_name + ' Start.')

    model_LinearRegression = linear_model.LinearRegression()  # 线性回归
    predict_linear = pre_model(model_LinearRegression, x_train, y_train, x_test)
    predict_linear = scaler_target.inverse_transform(predict_linear)
    flag_linear = deal_flag(predict_linear, minLen)
    accuracy_linear = deal_accuracy(y_rampflag, flag_linear)

    # rmse_linear = RMSE1(dataY, predict_linear)
    # mape_linear = MAPE1(dataY, predict_linear)
    # mae_linear = MAE1(dataY, predict_linear)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_linear: {}'.format(mae_linear)
    # eva_output += '\nrmse_linear: {}'.format(rmse_linear)
    # eva_output += '\nmape_linear: {}'.format(mape_linear)
    # eva_output += '\naccuracy_linear: {}'.format(accuracy_linear)
    # result_all.append(['lin', mae_linear, rmse_linear, mape_linear, accuracy_linear])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name, model_name_short, dataY, predict_linear, accuracy_linear)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_linear


def SVR(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'svr'
    model_name_short = 'svr'
    print(model_name + ' Start.')

    model_SVR = svm.SVR()  # SVR回归
    predict_svr = pre_model(model_SVR, x_train, y_train, x_test)
    predict_svr = scaler_target.inverse_transform(predict_svr)
    flag_svr = deal_flag(predict_svr, minLen)
    accuracy_svr = deal_accuracy(y_rampflag, flag_svr)

    # rmse_svr = RMSE1(dataY, predict_svr)
    # mape_svr = MAPE1(dataY, predict_svr)
    # mae_svr = MAE1(dataY, predict_svr)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_svr: {}'.format(mae_svr)
    # eva_output += '\nrmse_svr: {}'.format(rmse_svr)
    # eva_output += '\nmape_svr: {}'.format(mape_svr)
    # eva_output += '\naccuracy_svr: {}'.format(accuracy_svr)
    # result_all.append(['svr', mae_svr, rmse_svr, mape_svr, accuracy_svr])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name, model_name_short, dataY, predict_svr, accuracy_svr)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_svr


def KNN(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'kNeighbors'
    model_name_short = 'KNN'
    print(model_name + ' Start.')

    model_KNeighborsRegressor = neighbors.KNeighborsRegressor()  # KNN回归
    predict_kNeighbors = pre_model(model_KNeighborsRegressor, x_train, y_train, x_test)
    predict_kNeighbors = scaler_target.inverse_transform(predict_kNeighbors)
    flag_kNeighbors = deal_flag(predict_kNeighbors, minLen)
    accuracy_kNeighbors = deal_accuracy(y_rampflag, flag_kNeighbors)

    # rmse_kNeighbors = RMSE1(dataY, predict_kNeighbors)
    # mape_kNeighbors = MAPE1(dataY, predict_kNeighbors)
    # mae_kNeighbors = MAE1(dataY, predict_kNeighbors)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_kNeighbors: {}'.format(mae_kNeighbors)
    # eva_output += '\nrmse_kNeighbors: {}'.format(rmse_kNeighbors)
    # eva_output += '\nmape_kNeighbors: {}'.format(mape_kNeighbors)
    # eva_output += '\naccuracy_kNeighbors: {}'.format(accuracy_kNeighbors)
    # result_all.append(['KNN', mae_kNeighbors, rmse_kNeighbors, mape_kNeighbors, accuracy_kNeighbors])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name,
                                        model_name_short,
                                        dataY,
                                        predict_kNeighbors,
                                        accuracy_kNeighbors)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_kNeighbors


def MLP(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'mlp'
    model_name_short = 'mlp'
    print(model_name + ' Start.')

    model_MLP = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(20, 20, 20), random_state=2)  # MLP
    predict_mlp = pre_model(model_MLP, x_train, y_train, x_test)
    predict_mlp = scaler_target.inverse_transform(predict_mlp)
    flag_mlp = deal_flag(predict_mlp, minLen)
    accuracy_mlp = deal_accuracy(y_rampflag, flag_mlp)

    # rmse_mlp = RMSE1(dataY, predict_mlp)
    # mape_mlp = MAPE1(dataY, predict_mlp)
    # mae_mlp = MAE1(dataY, predict_mlp)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_mlp: {}'.format(mae_mlp)
    # eva_output += '\nrmse_mlp: {}'.format(rmse_mlp)
    # eva_output += '\nmape_mlp: {}'.format(mape_mlp)
    # eva_output += '\naccuracy_mlp: {}'.format(accuracy_mlp)
    # result_all.append(['mlp', mae_mlp, rmse_mlp, mape_mlp, accuracy_mlp])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name,
                                        model_name_short,
                                        dataY,
                                        predict_mlp,
                                        accuracy_mlp)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_mlp


def gradient_Boosting(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'gradientBoosting'
    model_name_short = 'grB'
    print(model_name + ' Start.')

    model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=50)  # GDBT
    predict_gradientBoosting = pre_model(model_GradientBoostingRegressor, x_train, y_train, x_test)
    predict_gradientBoosting = scaler_target.inverse_transform(predict_gradientBoosting)
    flag_gradientBoosting = deal_flag(predict_gradientBoosting, minLen)
    accuracy_gradientBoosting = deal_accuracy(y_rampflag, flag_gradientBoosting)

    # rmse_gradientBoosting = RMSE1(dataY, predict_gradientBoosting)
    # mape_gradientBoosting = MAPE1(dataY, predict_gradientBoosting)
    # mae_gradientBoosting = MAE1(dataY, predict_gradientBoosting)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_gradientBoosting: {}'.format(mae_gradientBoosting)
    # eva_output += '\nrmse_gradientBoosting: {}'.format(rmse_gradientBoosting)
    # eva_output += '\nmape_gradientBoosting: {}'.format(mape_gradientBoosting)
    # eva_output += '\naccuracy_gradientBoosting: {}'.format(accuracy_gradientBoosting)
    # result_all.append(
    #     ['grB', mae_gradientBoosting, rmse_gradientBoosting, mape_gradientBoosting, accuracy_gradientBoosting])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name,
                                        model_name_short,
                                        dataY,
                                        predict_gradientBoosting,
                                        accuracy_gradientBoosting)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_gradientBoosting


def extra_Tree(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'extraTree'
    model_name_short = 'eTr'
    print(model_name + ' Start.')

    model_ExtraTreeRegressor = ExtraTreeRegressor()  # extra tree
    predict_extraTree = pre_model(model_ExtraTreeRegressor, x_train, y_train, x_test)
    predict_extraTree = scaler_target.inverse_transform(predict_extraTree)
    flag_extraTree = deal_flag(predict_extraTree, minLen)
    accuracy_extraTree = deal_accuracy(y_rampflag, flag_extraTree)

    # rmse_extraTree = RMSE1(dataY, predict_extraTree)
    # mape_extraTree = MAPE1(dataY, predict_extraTree)
    # mae_extraTree = MAE1(dataY, predict_extraTree)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_extraTree: {}'.format(mae_extraTree)
    # eva_output += '\nrmse_extraTree: {}'.format(rmse_extraTree)
    # eva_output += '\nmape_extraTree: {}'.format(mape_extraTree)
    # eva_output += '\naccuracy_extraTree: {}'.format(accuracy_extraTree)
    # result_all.append(['eTr', mae_extraTree, rmse_extraTree, mape_extraTree, accuracy_extraTree])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name,
                                        model_name_short,
                                        dataY,
                                        predict_extraTree,
                                        accuracy_extraTree)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_extraTree


def bagging(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'bagging'
    model_name_short = 'bag'
    print(model_name + ' Start.')

    model_BaggingRegressor = BaggingRegressor()  # bggingRegressor
    predict_bagging = pre_model(model_BaggingRegressor, x_train, y_train, x_test)
    predict_bagging = scaler_target.inverse_transform(predict_bagging)
    flag_bagging = deal_flag(predict_bagging, minLen)
    accuracy_bagging = deal_accuracy(y_rampflag, flag_bagging)

    # rmse_bagging = RMSE1(dataY, predict_bagging)
    # mape_bagging = MAPE1(dataY, predict_bagging)
    # mae_bagging = MAE1(dataY, predict_bagging)
    #
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_bagging: {}'.format(mae_bagging)
    # eva_output += '\nrmse_bagging: {}'.format(rmse_bagging)
    # eva_output += '\nmape_bagging: {}'.format(mape_bagging)
    # eva_output += '\naccuracy_bagging: {}'.format(accuracy_bagging)
    #
    # result_all.append(['bag', mae_bagging, rmse_bagging, mape_bagging, accuracy_bagging])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name,
                                        model_name_short,
                                        dataY,
                                        predict_bagging,
                                        accuracy_bagging)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_bagging


def adaboost(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'adaboost'
    model_name_short = 'ada'
    print(model_name + ' Start.')

    model_AdaboostRegressor = AdaBoostRegressor()  # adaboostRegressor
    predict_adaboost = pre_model(model_AdaboostRegressor, x_train, y_train, x_test)
    predict_adaboost = scaler_target.inverse_transform(predict_adaboost)
    flag_adaboost = deal_flag(predict_adaboost, minLen)
    accuracy_adaboost = deal_accuracy(y_rampflag, flag_adaboost)

    # rmse_adaboost = RMSE1(dataY, predict_adaboost)
    # mape_adaboost = MAPE1(dataY, predict_adaboost)
    # mae_adaboost = MAE1(dataY, predict_adaboost)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_adaboost: {}'.format(mae_adaboost)
    # eva_output += '\nrmse_adaboost: {}'.format(rmse_adaboost)
    # eva_output += '\nmape_adaboost: {}'.format(mape_adaboost)
    # eva_output += '\naccuracy_adaboost: {}'.format(accuracy_adaboost)
    #
    # result_all.append(['ada', mae_adaboost, rmse_adaboost, mape_adaboost, accuracy_adaboost])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name,
                                        model_name_short,
                                        dataY,
                                        predict_adaboost,
                                        accuracy_adaboost)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_adaboost


def LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'LSTM'
    model_name_short = 'LSTM'
    print(model_name + ' Start.')

    feature_num = x_train.shape[1]
    x_train_lstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_lstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Modelling and Predicting
    print('Build model...')

    model = build_LSTM(feature_num)
    # model = buildOLSTM(feature_num)

    global show_process
    time_callback = TimeHistory()
    history = model.fit(x_train_lstm, y_train, epochs=16, validation_split=0.05, verbose=show_process,
                        callbacks=[time_callback])
    train_time = time_callback.totaltime

    predict_lstm = model.predict(x_test_lstm)

    predict_lstm = predict_lstm.reshape(-1, )
    predict_lstm = scaler_target.inverse_transform(predict_lstm)
    flag_lstm = deal_flag(predict_lstm, minLen)
    accuracy_lstm = deal_accuracy(y_rampflag, flag_lstm)

    # rmse_nlstm = RMSE1(dataY, predict_nlstm)
    # mape_nlstm = MAPE1(dataY, predict_nlstm)
    # mae_nlstm = MAE1(dataY, predict_nlstm)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_nlstm: {}'.format(mae_nlstm)
    # eva_output += '\nrmse_nlstm: {}'.format(rmse_nlstm)
    # eva_output += '\nmape_nlstm: {}'.format(mape_nlstm)
    # eva_output += '\naccuracy_nlstm: {}'.format(accuracy_nlstm)
    # result_all.append(['nlstm', mae_nlstm, rmse_nlstm, mape_nlstm, accuracy_nlstm])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_lstm,
                                           accuracy_lstm,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_lstm


def EMD_LSTM(emd_list, y_rampflag, dataY, emd_decay_rate):
    model_name = 'EMD'
    model_name_short = 'EMD'
    print(model_name + ' Start.')

    model = build_LSTM(ahead_num)

    global show_process

    predict_emd_list = []
    train_time_total = 0
    for ie in range(len(emd_list)):
        # emd_list[ie][0] = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        # emd_list[ie][2] = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trX = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        EMD_teX = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trY = emd_list[ie][1]

        time_callback = TimeHistory()
        model.fit(EMD_trX, EMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time
        # model.fit(emd_list[ie][0], emd_list[ie][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        predict_emd_part = model.predict(EMD_teX)
        predict_emd_part = predict_emd_part.reshape(-1, )
        predict_emd_list.append(predict_emd_part)
        print('EMD_imf ' + str(ie + 1) + ' Complete.')

    # wavefun = pywt.Wavelet('db1')
    # predict_EN = iswt_decom(predict_emd_list, wavefun)

    predict_EMD = predict_emd_list[-1]
    for i in range(len(predict_emd_list) - 1):
        predict_EMD = predict_EMD + predict_emd_list[i] * emd_decay_rate
    # predict_noise = predict_EN - predict_emd_list[-1]

    predict_EMD = scaler_target.inverse_transform(predict_EMD)
    flag_EMD = deal_flag(predict_EMD, minLen)
    accuracy_EMD = deal_accuracy(y_rampflag, flag_EMD)

    # rmse_EN = RMSE1(dataY, predict_EN)
    # mape_EN = MAPE1(dataY, predict_EN)
    # mae_EN = MAE1(dataY, predict_EN)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_EN: {}'.format(mae_EN)
    # eva_output += '\nrmse_EN: {}'.format(rmse_EN)
    # eva_output += '\nmape_EN: {}'.format(mape_EN)
    # eva_output += '\naccuracy_EN: {}'.format(accuracy_EN)
    # result_all.append(['EN', mae_EN, rmse_EN, mape_EN, accuracy_EN])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_EMD,
                                           accuracy_EMD,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_EMD


def Wavelet_LSTM(wvlt_list, y_rampflag, dataY):
    model_name = 'Wvlt'
    model_name_short = 'Wvlt'
    print(model_name + ' Start.')

    model = build_LSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_Wvlt_list = []
    for i_wvlt in range(len(wvlt_list)):
        wvlt_trX = np.reshape(wvlt_list[i_wvlt][0],
                              (wvlt_list[i_wvlt][0].shape[0],
                               wvlt_list[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list[i_wvlt][2],
                              (wvlt_list[i_wvlt][2].shape[0],
                               wvlt_list[i_wvlt][2].shape[1], 1))
        wvlt_trY = wvlt_list[i_wvlt][1]

        time_callback = TimeHistory()
        model.fit(wvlt_trX, wvlt_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        # model.fit(wvlt_list[i_wvlt][0], wvlt_list[i_wvlt][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_Wvlt = model.predict(wvlt_teX)
        predict_Wvlt = predict_Wvlt.reshape(-1, )
        predict_Wvlt_list.append(predict_Wvlt)
        print('wvlt_level ' + str(i_wvlt + 1) + ' Complete.')

    wavefun = pywt.Wavelet('db1')

    predict_Wvlt = iswt_decom(predict_Wvlt_list, wavefun)
    predict_Wvlt = scaler_target.inverse_transform(predict_Wvlt)
    flag_Wvlt = deal_flag(predict_Wvlt, minLen)
    accuracy_Wvlt = deal_accuracy(y_rampflag, flag_Wvlt)

    # # dataY1 = scaler_target.inverse_transform(dataY1)
    # # dataY1 = dataY1.T.tolist()[0]
    # rmse_WN = RMSE1(dataY, predict_WN)
    # mape_WN = MAPE1(dataY, predict_WN)
    # mae_WN = MAE1(dataY, predict_WN)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_WN: {}'.format(mae_WN)
    # eva_output += '\nrmse_WN: {}'.format(rmse_WN)
    # eva_output += '\nmape_WN: {}'.format(mape_WN)
    # eva_output += '\naccuracy_WN: {}'.format(accuracy_WN)
    # result_all.append(['WN', mae_WN, rmse_WN, mape_WN, accuracy_WN])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_Wvlt,
                                           accuracy_Wvlt,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_Wvlt


def Nested_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'NLSTM'
    model_name_short = 'NLSTM'
    print('NLSTM Begin.')

    feature_num = x_train.shape[1]
    x_train_nlstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_nlstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Modelling and Predicting
    print('Build model...')

    model = buildNLSTM(feature_num)
    # model = buildOLSTM(feature_num)

    global show_process

    time_callback = TimeHistory()
    history = model.fit(x_train_nlstm, y_train, epochs=16, validation_split=0.05, verbose=show_process,
                        callbacks=[time_callback])
    train_time = time_callback.totaltime

    predict_nlstm = model.predict(x_test_nlstm)

    # history = model.fit(x_train, y_train, epochs=16, validation_split=0.05, verbose=2)
    #
    # predict_nlstm = model.predict(x_test)
    predict_nlstm = predict_nlstm.reshape(-1, )
    predict_nlstm = scaler_target.inverse_transform(predict_nlstm)
    flag_nlstm = deal_flag(predict_nlstm, minLen)
    accuracy_nlstm = deal_accuracy(y_rampflag, flag_nlstm)

    # rmse_nlstm = RMSE1(dataY, predict_nlstm)
    # mape_nlstm = MAPE1(dataY, predict_nlstm)
    # mae_nlstm = MAE1(dataY, predict_nlstm)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_nlstm: {}'.format(mae_nlstm)
    # eva_output += '\nrmse_nlstm: {}'.format(rmse_nlstm)
    # eva_output += '\nmape_nlstm: {}'.format(mape_nlstm)
    # eva_output += '\naccuracy_nlstm: {}'.format(accuracy_nlstm)
    # result_all.append(['nlstm', mae_nlstm, rmse_nlstm, mape_nlstm, accuracy_nlstm])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_nlstm,
                                           accuracy_nlstm,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    print('NLSTM Complete.')
    return predict_nlstm


def EMD_NLSTM(emd_list, y_rampflag, dataY, emd_decay_rate):
    model_name = 'EN'
    model_name_short = 'EN'
    print(model_name + ' Start.')

    model = buildNLSTM(ahead_num)

    global show_process

    predict_emd_list = []
    train_time_total = 0
    for ie in range(len(emd_list)):
        # emd_list[ie][0] = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        # emd_list[ie][2] = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trX = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        EMD_teX = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trY = emd_list[ie][1]
        time_callback = TimeHistory()
        model.fit(EMD_trX, EMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time
        # model.fit(emd_list[ie][0], emd_list[ie][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        predict_emd_part = model.predict(EMD_teX)
        predict_emd_part = predict_emd_part.reshape(-1, )
        predict_emd_list.append(predict_emd_part)
        print('EMD_imf ' + str(ie + 1) + ' Complete.')

    # wavefun = pywt.Wavelet('db1')
    # predict_EN = iswt_decom(predict_emd_list, wavefun)

    predict_EN = predict_emd_list[-1]
    for i in range(len(predict_emd_list) - 1):
        predict_EN = predict_EN + predict_emd_list[i] * emd_decay_rate
    # predict_noise = predict_EN - predict_emd_list[-1]

    predict_EN = scaler_target.inverse_transform(predict_EN)
    flag_EN = deal_flag(predict_EN, minLen)
    accuracy_EN = deal_accuracy(y_rampflag, flag_EN)

    # rmse_EN = RMSE1(dataY, predict_EN)
    # mape_EN = MAPE1(dataY, predict_EN)
    # mae_EN = MAE1(dataY, predict_EN)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_EN: {}'.format(mae_EN)
    # eva_output += '\nrmse_EN: {}'.format(rmse_EN)
    # eva_output += '\nmape_EN: {}'.format(mape_EN)
    # eva_output += '\naccuracy_EN: {}'.format(accuracy_EN)
    # result_all.append(['EN', mae_EN, rmse_EN, mape_EN, accuracy_EN])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_EN,
                                           accuracy_EN,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_EN


def Wavelet_NLSTM(wvlt_list, y_rampflag, dataY):
    model_name = 'WN'
    model_name_short = 'WN'
    print(model_name + ' Start.')

    model = buildNLSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_WN_list = []
    for i_wvlt in range(len(wvlt_list)):
        wvlt_trX = np.reshape(wvlt_list[i_wvlt][0],
                              (wvlt_list[i_wvlt][0].shape[0],
                               wvlt_list[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list[i_wvlt][2],
                              (wvlt_list[i_wvlt][2].shape[0],
                               wvlt_list[i_wvlt][2].shape[1], 1))
        wvlt_trY = wvlt_list[i_wvlt][1]

        time_callback = TimeHistory()
        model.fit(wvlt_trX, wvlt_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        # model.fit(wvlt_list[i_wvlt][0], wvlt_list[i_wvlt][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_WN = model.predict(wvlt_teX)
        predict_WN = predict_WN.reshape(-1, )
        predict_WN_list.append(predict_WN)
        print('wvlt_level ' + str(i_wvlt + 1) + ' Complete.')

    wavefun = pywt.Wavelet('db1')

    predict_WN = iswt_decom(predict_WN_list, wavefun)
    predict_WN = scaler_target.inverse_transform(predict_WN)
    flag_WN = deal_flag(predict_WN, minLen)
    accuracy_WN = deal_accuracy(y_rampflag, flag_WN)

    # # dataY1 = scaler_target.inverse_transform(dataY1)
    # # dataY1 = dataY1.T.tolist()[0]
    # rmse_WN = RMSE1(dataY, predict_WN)
    # mape_WN = MAPE1(dataY, predict_WN)
    # mae_WN = MAE1(dataY, predict_WN)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_WN: {}'.format(mae_WN)
    # eva_output += '\nrmse_WN: {}'.format(rmse_WN)
    # eva_output += '\nmape_WN: {}'.format(mape_WN)
    # eva_output += '\naccuracy_WN: {}'.format(accuracy_WN)
    # result_all.append(['WN', mae_WN, rmse_WN, mape_WN, accuracy_WN])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_WN,
                                           accuracy_WN,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_WN


def Stacked_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'SLSTM'
    model_name_short = 'SLSTM'
    print(model_name + ' Start.')

    feature_num = x_train.shape[1]
    x_train_slstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_slstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Modelling and Predicting
    print('Build model...')

    model = buildSLSTM(feature_num)
    # model = buildOLSTM(feature_num)

    global show_process

    time_callback = TimeHistory()
    history = model.fit(x_train_slstm, y_train, epochs=16, validation_split=0.05, verbose=show_process,
                        callbacks=[time_callback])
    train_time = time_callback.totaltime

    predict_slstm = model.predict(x_test_slstm)

    # history = model.fit(x_train, y_train, epochs=16, validation_split=0.05, verbose=2)
    #
    # predict_slstm = model.predict(x_test)
    predict_slstm = predict_slstm.reshape(-1, )
    predict_slstm = scaler_target.inverse_transform(predict_slstm)
    flag_slstm = deal_flag(predict_slstm, minLen)
    accuracy_slstm = deal_accuracy(y_rampflag, flag_slstm)

    # rmse_slstm = RMSE1(dataY, predict_slstm)
    # mape_slstm = MAPE1(dataY, predict_slstm)
    # mae_slstm = MAE1(dataY, predict_slstm)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_slstm: {}'.format(mae_slstm)
    # eva_output += '\nrmse_slstm: {}'.format(rmse_slstm)
    # eva_output += '\nmape_slstm: {}'.format(mape_slstm)
    # eva_output += '\naccuracy_slstm: {}'.format(accuracy_slstm)
    # result_all.append(['slstm', mae_slstm, rmse_slstm, mape_slstm, accuracy_slstm])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_slstm,
                                           accuracy_slstm,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_slstm


def EMD_SLSTM(emd_list, y_rampflag, dataY, emd_decay_rate):
    model_name = 'ES'
    model_name_short = 'ES'
    print(model_name + ' Start.')

    model = buildSLSTM(ahead_num)

    global show_process

    predict_emd_list = []
    train_time_total = 0
    for ie in range(len(emd_list)):
        # emd_list[ie][0] = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        # emd_list[ie][2] = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trX = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        EMD_teX = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trY = emd_list[ie][1]
        time_callback = TimeHistory()
        model.fit(EMD_trX, EMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time
        # model.fit(emd_list[ie][0], emd_list[ie][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        predict_emd_part = model.predict(EMD_teX)
        predict_emd_part = predict_emd_part.reshape(-1, )
        predict_emd_list.append(predict_emd_part)
        print('EMD_imf ' + str(ie + 1) + ' Complete.')

    # wavefun = pywt.Wavelet('db1')
    # predict_ES = iswt_decom(predict_emd_list, wavefun)

    predict_ES = predict_emd_list[-1]
    for i in range(len(predict_emd_list) - 1):
        predict_ES = predict_ES + predict_emd_list[i] * emd_decay_rate
    # predict_noise = predict_ES - predict_emd_list[-1]

    predict_ES = scaler_target.inverse_transform(predict_ES)
    flag_ES = deal_flag(predict_ES, minLen)
    accuracy_ES = deal_accuracy(y_rampflag, flag_ES)

    # rmse_ES = RMSE1(dataY, predict_ES)
    # mape_ES = MAPE1(dataY, predict_ES)
    # mae_ES = MAE1(dataY, predict_ES)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_ES: {}'.format(mae_ES)
    # eva_output += '\nrmse_ES: {}'.format(rmse_ES)
    # eva_output += '\nmape_ES: {}'.format(mape_ES)
    # eva_output += '\naccuracy_ES: {}'.format(accuracy_ES)
    # result_all.append(['EN', mae_ES, rmse_ES, mape_ES, accuracy_ES])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_ES,
                                           accuracy_ES,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_ES


def Wavelet_SLSTM(wvlt_list, y_rampflag, dataY):
    model_name = 'WS'
    model_name_short = 'WS'
    print(model_name + ' Start.')

    model = buildSLSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_WS_list = []
    for i_wvlt in range(len(wvlt_list)):
        wvlt_trX = np.reshape(wvlt_list[i_wvlt][0],
                              (wvlt_list[i_wvlt][0].shape[0],
                               wvlt_list[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list[i_wvlt][2],
                              (wvlt_list[i_wvlt][2].shape[0],
                               wvlt_list[i_wvlt][2].shape[1], 1))
        wvlt_trY = wvlt_list[i_wvlt][1]

        time_callback = TimeHistory()
        model.fit(wvlt_trX, wvlt_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        # model.fit(wvlt_list[i_wvlt][0], wvlt_list[i_wvlt][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_WS = model.predict(wvlt_teX)
        predict_WS = predict_WS.reshape(-1, )
        predict_WS_list.append(predict_WS)
        print('wvlt_level ' + str(i_wvlt) + ' Complete.')

    wavefun = pywt.Wavelet('db1')

    predict_WS = iswt_decom(predict_WS_list, wavefun)
    predict_WS = scaler_target.inverse_transform(predict_WS)
    flag_WS = deal_flag(predict_WS, minLen)
    accuracy_WS = deal_accuracy(y_rampflag, flag_WS)

    # # dataY1 = scaler_target.inverse_transform(dataY1)
    # # dataY1 = dataY1.T.tolist()[0]
    # rmse_WS = RMSE1(dataY, predict_WS)
    # mape_WS = MAPE1(dataY, predict_WS)
    # mae_WS = MAE1(dataY, predict_WS)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_WS: {}'.format(mae_WS)
    # eva_output += '\nrmse_WS: {}'.format(rmse_WS)
    # eva_output += '\nmape_WS: {}'.format(mape_WS)
    # eva_output += '\naccuracy_WS: {}'.format(accuracy_WS)
    # result_all.append(['WN', mae_WS, rmse_WS, mape_WS, accuracy_WS])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_WS,
                                           accuracy_WS,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_WS


def BLSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'BLSTM'
    model_name_short = 'BLSTM'
    print(model_name + ' Start.')

    feature_num = x_train.shape[1]
    x_train_blstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_blstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Modelling and Predicting
    print('Build model...')

    model = buildBLSTM(feature_num)
    # model = buildOLSTM(feature_num)

    global show_process

    time_callback = TimeHistory()
    history = model.fit(x_train_blstm, y_train, epochs=16, validation_split=0.05, verbose=show_process,
                        callbacks=[time_callback])
    train_time = time_callback.totaltime

    predict_blstm = model.predict(x_test_blstm)

    # history = model.fit(x_train, y_train, epochs=16, validation_split=0.05, verbose=2)
    #
    # predict_blstm = model.predict(x_test)
    predict_blstm = predict_blstm.reshape(-1, )
    predict_blstm = scaler_target.inverse_transform(predict_blstm)
    flag_blstm = deal_flag(predict_blstm, minLen)
    accuracy_blstm = deal_accuracy(y_rampflag, flag_blstm)

    # rmse_blstm = RMSE1(dataY, predict_blstm)
    # mape_blstm = MAPE1(dataY, predict_blstm)
    # mae_blstm = MAE1(dataY, predict_blstm)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_blstm: {}'.format(mae_blstm)
    # eva_output += '\nrmse_blstm: {}'.format(rmse_blstm)
    # eva_output += '\nmape_blstm: {}'.format(mape_blstm)
    # eva_output += '\naccuracy_blstm: {}'.format(accuracy_blstm)
    # result_all.append(['blstm', mae_blstm, rmse_blstm, mape_blstm, accuracy_blstm])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_blstm,
                                           accuracy_blstm,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_blstm


def EMD_BLSTM(emd_list, y_rampflag, dataY, emd_decay_rate):
    model_name = 'EB'
    model_name_short = 'EB'
    print(model_name + ' Start.')

    model = buildBLSTM(ahead_num)

    global show_process

    predict_emd_list = []
    train_time_total = 0
    for ie in range(len(emd_list)):
        # emd_list[ie][0] = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        # emd_list[ie][2] = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trX = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        EMD_teX = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trY = emd_list[ie][1]
        time_callback = TimeHistory()
        model.fit(EMD_trX, EMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time
        # model.fit(emd_list[ie][0], emd_list[ie][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        predict_emd_part = model.predict(EMD_teX)
        predict_emd_part = predict_emd_part.reshape(-1, )
        predict_emd_list.append(predict_emd_part)
        print('EMD_imf ' + str(ie + 1) + ' Complete.')

    # wavefun = pywt.Wavelet('db1')
    # predict_EB = iswt_decom(predict_emd_list, wavefun)

    predict_EB = predict_emd_list[-1]
    for i in range(len(predict_emd_list) - 1):
        predict_EB = predict_EB + predict_emd_list[i] * emd_decay_rate
    # predict_noise = predict_EB - predict_emd_list[-1]

    predict_EB = scaler_target.inverse_transform(predict_EB)
    flag_EB = deal_flag(predict_EB, minLen)
    accuracy_EB = deal_accuracy(y_rampflag, flag_EB)

    # rmse_EB = RMSE1(dataY, predict_EB)
    # mape_EB = MAPE1(dataY, predict_EB)
    # mae_EB = MAE1(dataY, predict_EB)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_EB: {}'.format(mae_EB)
    # eva_output += '\nrmse_EB: {}'.format(rmse_EB)
    # eva_output += '\nmape_EB: {}'.format(mape_EB)
    # eva_output += '\naccuracy_EB: {}'.format(accuracy_EB)
    # result_all.append(['EN', mae_EB, rmse_EB, mape_EB, accuracy_EB])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_EB,
                                           accuracy_EB,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_EB


def Wavelet_BLSTM(wvlt_list, y_rampflag, dataY):
    model_name = 'WB'
    model_name_short = 'WB'
    print(model_name + ' Start.')

    model = buildBLSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_WB_list = []
    for i_wvlt in range(len(wvlt_list)):
        wvlt_trX = np.reshape(wvlt_list[i_wvlt][0],
                              (wvlt_list[i_wvlt][0].shape[0],
                               wvlt_list[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list[i_wvlt][2],
                              (wvlt_list[i_wvlt][2].shape[0],
                               wvlt_list[i_wvlt][2].shape[1], 1))
        wvlt_trY = wvlt_list[i_wvlt][1]

        time_callback = TimeHistory()
        model.fit(wvlt_trX, wvlt_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        # model.fit(wvlt_list[i_wvlt][0], wvlt_list[i_wvlt][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_WB = model.predict(wvlt_teX)
        predict_WB = predict_WB.reshape(-1, )
        predict_WB_list.append(predict_WB)
        print('wvlt_level ' + str(i_wvlt) + ' Complete.')

    wavefun = pywt.Wavelet('db1')

    predict_WB = iswt_decom(predict_WB_list, wavefun)
    predict_WB = scaler_target.inverse_transform(predict_WB)
    flag_WB = deal_flag(predict_WB, minLen)
    accuracy_WB = deal_accuracy(y_rampflag, flag_WB)

    # # dataY1 = scaler_target.inverse_transform(dataY1)
    # # dataY1 = dataY1.T.tolist()[0]
    # rmse_WB = RMSE1(dataY, predict_WB)
    # mape_WB = MAPE1(dataY, predict_WB)
    # mae_WB = MAE1(dataY, predict_WB)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_WB: {}'.format(mae_WB)
    # eva_output += '\nrmse_WB: {}'.format(rmse_WB)
    # eva_output += '\nmape_WB: {}'.format(mape_WB)
    # eva_output += '\naccuracy_WB: {}'.format(accuracy_WB)
    # result_all.append(['WN', mae_WB, rmse_WB, mape_WB, accuracy_WB])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_WB,
                                           accuracy_WB,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_WB

#########################################################################


def main(start_num, interval_ori):

    # random_state = np.random.RandomState(7)
    np.random.RandomState(7)

    #########################################################################

    # lookback number
    global ahead_num
    ahead_num = 4

    global interval
    interval = interval_ori

    global minLen
    minLen = 0

    # 1 for features; 2 for timeseries; 3 for feature & timeseries.
    global x_mode
    x_mode = 2

    global wvlt_lv
    wvlt_lv = 3

    global show_process
    show_process = 1

    #########################################################################

    num = 12
    filename1 = "dataset\\PRSA_Data_"
    filename2 = ".csv"
    filename = [filename1, filename2]

    # training number
    startNum = start_num
    trainNum = (24 * 1000) // interval
    testNum = (24 * 20) // interval

    emd_decay_rate = 1.00

    #########################################################################

    # dataset_list = read_csv_all(filename, trainNum, testNum, startNum, num, interval)
    # dataset = dataset_list[0]      #PM2
    # dataset = dataset_list[1]      #PM10
    # dataset = dataset_list[2]      #SO2
    # dataset = dataset_list[3]      #NO2
    # dataset = dataset_list[4]      #CO
    # dataset = dataset_list[5]      #O3
    # dataset = dataset_list[6]      #TEMP
    # dataset = dataset_list[7]      #PRES
    # dataset = dataset_list[8]      #DEWP

    dataset = read_csv_PM2(filename, trainNum, testNum, startNum, num, interval)
    # dataset = read_csv_PM10(filename, trainNum, testNum, startNum, num, interval)
    # dataset = read_csv_SO2(filename, trainNum, testNum, startNum, num, interval)
    # dataset = read_csv_NO2(filename, trainNum, testNum, startNum, num, interval)
    # dataset = read_csv_CO(filename, trainNum, testNum, startNum, num, interval)
    # dataset = read_csv_O3(filename, trainNum, testNum, startNum, num, interval)
    # dataset = read_csv_TEMP(filename, trainNum, testNum, startNum, num, interval)
    # dataset = read_csv_PRES(filename, trainNum, testNum, startNum, num, interval)
    # dataset = read_csv_DEWP(filename, trainNum, testNum, startNum, num, interval)

    #########################################################################

    x_train, y_train, x_test, y_test = load_data_ts(trainNum, testNum, startNum, dataset)
    emd_list = load_data_emd(trainNum, testNum, startNum, dataset)
    wvlt_list, _ = load_data_wvlt(trainNum, testNum, startNum, dataset)

    #########################################################################

    # #####culculate Accuracy by rampflag
    global scaler_target
    dataY = scaler_target.inverse_transform(y_test)
    # minLen = np.mean(dataY) * 0.25
    minLen = 0
    print('Accuracy Flag:', minLen)
    y_rampflag = deal_flag(dataY, minLen)

    ######=========================Modelling and Predicting=========================#####
    print("======================================================")
    global eva_output, result_all
    eva_output = '\nEvaluation.'
    result_all = []

    predict_decideTree = Decide_Tree(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    predict_randomForest = Random_forest(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    predict_linear = Linear_Regression(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    predict_svr = SVR(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    predict_kNeighbors = KNN(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    predict_mlp = MLP(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    predict_gradientBoosting = gradient_Boosting(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    predict_extraTree = extra_Tree(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    predict_bagging = bagging(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    predict_adaboost = adaboost(x_train, y_train, x_test, y_test, y_rampflag, dataY)

    predict_lstm = LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    predict_lstm_emd = EMD_LSTM(emd_list, y_rampflag, dataY, emd_decay_rate)
    predict_wavelet = Wavelet_LSTM(wvlt_list, y_rampflag, dataY)

    predict_nlstm = Nested_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    predict_EN = EMD_NLSTM(emd_list, y_rampflag, dataY, emd_decay_rate)
    predict_WN = Wavelet_NLSTM(wvlt_list, y_rampflag, dataY)

    predict_slstm = Stacked_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    predict_ES = EMD_SLSTM(emd_list, y_rampflag, dataY, emd_decay_rate)
    predict_WS = Wavelet_SLSTM(wvlt_list, y_rampflag, dataY)

    predict_blstm = BLSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    predict_EB = EMD_BLSTM(emd_list, y_rampflag, dataY, emd_decay_rate)
    predict_WB = Wavelet_BLSTM(wvlt_list, y_rampflag, dataY)

    print(eva_output)

    # # print(result_all)
    save_file_name = "result\\dataset_all_result.csv"

    csv_file = open(save_file_name, "w", newline="")  # 创建csv文件
    writer = csv.writer(csv_file)  # 创建写的对象
    writer.writerow(["", "MAE", "RMSE", "MAPE", "R2", "ACC", "TIME"])
    for wtr in range(len(result_all)):
        writer.writerow(result_all[wtr])
        
    # backup for Denied permission.
    save_file_name = "result\\dataset_all_result_backup.csv"

    csv_file = open(save_file_name, "w", newline="")  # 创建csv文件
    writer = csv.writer(csv_file)  # 创建写的对象
    writer.writerow(["", "MAE", "RMSE", "MAPE", "R2", "ACC", "TIME"])
    for wtr in range(len(result_all)):
        writer.writerow(result_all[wtr])

    # ###===============画图===========================
    # # fontsize = 18
    # main_linewidth = 2
    # second_linewidth = 1.5
    #
    # plt.figure(1, figsize=(15, 5))
    # plt.plot(dataY, "black", label="true", linewidth=main_linewidth)
    # # plt.plot(predict_EN, "pink", label="EN", linewidth=second_linewidth)
    # # plt.plot(predict_WN, "red", label="WN", linewidth=second_linewidth)
    # # plt.plot(predict_MV, "red", label="MV", linewidth=second_linewidth)
    # # plt.plot(predict_WNb, "blue", label="Nbeats", linewidth=second_linewidth)
    # # plt.plot(predict_EN, "red", label="EMD+LSTM", linewidth=second_linewidth)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title(str(testNum)+" Hours")
    # plt.legend(loc='best')
    # plt.show()
    #
    # print('\nComplete.')

    return result_all


if __name__ == "__main__":

    start_num = 0
    interval_ori = 1
    _ = main(start_num, interval_ori)




