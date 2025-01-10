import os
import time
import numpy as np
import scipy.io
import tensorflow as tf
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, stations, result_file, Tn,train_test_size=0.3, adam_epochs=1000, lstmlayers=120, alphas=[1, 0.2], directory='data/',seed=42):
        self.adam_epochs = adam_epochs
        self.lstmlayers = lstmlayers
        self.Tn = Tn
        self.alphas = alphas
        self.train_test_size = train_test_size
        self.result_file = result_file
        self.directory = directory
        self.stations = stations
        self.log_vars = tf.Variable(tf.zeros((6,),dtype=tf.float64), trainable=True)
        self.seed = seed
        # self.set_seed(seed)

    def load_data(self, station):
        Dir = self.directory + f'{station}_spectra.mat'
        mat = scipy.io.loadmat(Dir)
        return mat
    # def set_seed(self,seed):
    #     np.random.seed(seed)
    #     tf.random.set_seed(seed)
    #     os.environ['PYTHONHASHSEED'] = str(seed)
    def custom_loss(self, Y_true, Y_pred, max_Y, f1, f2, f3, alpha):
        loss_data1 = tf.reduce_mean(tf.square(Y_pred[:, :, 0] - Y_true[:, :, 0]))
        loss_data2 = tf.reduce_mean(tf.square(Y_pred[:, :, 1] - Y_true[:, :, 1]))
        loss_data3 = tf.reduce_mean(tf.square(Y_pred[:, :, 2] - Y_true[:, :, 2]))
        loss_phy1 = tf.reduce_mean(tf.square(Y_pred[:, :, 0] * max_Y[:, :, 0] - f1[:, :, 0] * Y_pred[:, :, 1] * max_Y[:, :, 1]))
        loss_phy3 = tf.reduce_mean(tf.square(Y_pred[:, :, 2] * max_Y[:, :, 2] - max_Y[:, :, 0] * Y_pred[:, :, 0] / f3[:, :, 0]))
        losses = tf.convert_to_tensor([loss_data1, loss_data2, loss_data3, loss_phy1, loss_phy3])

        precision = tf.exp(-self.log_vars)
        weighted_losses = precision * losses + self.log_vars

        total_loss = tf.reduce_sum(weighted_losses)
        return total_loss, loss_data1, loss_data2, loss_data3, loss_phy1, loss_phy3, alpha, alpha, alpha

    def build_model(self, input_shape):
        layers = [
            tf.keras.Input(shape=(None, input_shape)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self.lstmlayers, return_sequences=True)),
            tf.keras.layers.Dense(3, activation=tf.nn.softplus)
        ]
        model = tf.keras.models.Sequential(layers)
        model.build()
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, decay=1e-4),
                      loss=self.custom_loss)
        return model

    def calculate_statistics(self, y_true, y_pred):
        statistics = {
            'rmse': lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'mae': lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
            'r2': lambda y_true, y_pred: np.array([r2_score(y_true[i], y_pred[i]) for i in range(len(y_true))])
        }
        results = {name: func(y_true, y_pred) for name, func in statistics.items()}
        return results

    def train_model(self):
        for station in self.stations:
            for alpha in self.alphas:
                mat = self.load_data(station)
                tt = time.strftime("%Y%m%d_%H%M%S")
                result_file_name = self.result_file + f'{alpha}_result/{station}_result/Phy-BiLSTM/phy-BiLSTM_batch_lstmlayers_{self.lstmlayers}_epoch_{self.adam_epochs}_time{tt}/'
                os.makedirs(result_file_name, exist_ok=True)
                up_psa, down_psa = mat['up_sa'][:,:,:], mat['down_sa'][:,:,:]
                up_psv, down_psv = mat['up_sv'][:,:,:], mat['down_sv'][:,:,:]
                up_sd, down_sd = mat['up_sd'][:,:,:], mat['down_sd'][:,:,:]
                up_psa_pred, up_psv_pred, up_sd_pred = mat['up_sa_pred'][:,:,:], mat['up_sv_pred'][:,:,:], mat['up_sd_pred'][:,:,:]
                f1, f2, f3 = up_psa_pred / up_psv_pred, up_psv_pred / up_sd_pred, up_psa_pred / up_sd_pred
                hh = ShuffleSplit(n_splits=1, test_size=200 / len(down_psa), random_state=1)
                _, index = next(hh.split(down_psa))
                up_psa_true, down_psa_true = up_psa[index], down_psa[index]
                up_psv_true, down_psv_true = up_psv[index], down_psv[index]
                up_sd_true, down_sd_true = up_sd[index], down_sd[index]
                f1_true, f2_true, f3_true = f1[index], f2[index], f3[index]
                train_index, test_index = train_test_split(range(len(down_psa_true)), test_size=self.train_test_size, random_state=1)
                up_true, down_true = np.concatenate((up_psa_true, up_psv_true, up_sd_true), axis=-1), np.concatenate(
                    (down_psa_true, down_psv_true, down_sd_true), axis=-1)
                Y_data = up_true
                X_data = down_true
                max_X_data = np.max(X_data, axis=1).reshape(X_data.shape[0], 1, 3)
                max_Y_data = np.max(Y_data, axis=1).reshape(Y_data.shape[0], 1, 3)
                X_data = X_data / max_X_data
                Y_data = Y_data / max_Y_data
                X_train_data, X_test_data, Y_train_data, Y_test_data = X_data[train_index],X_data[test_index],Y_data[train_index],Y_data[test_index]
                f1_train, f2_train, f3_train = f1_true[train_index], f2_true[train_index], f3_true[train_index]
                max_Y_train = max_Y_data[train_index]
                model = self.build_model(X_train_data.shape[-1])
                best_val_loss, best_train_loss = float('inf'), float('inf')


                start_time = time.time()
                tim = time.time()
                total_loss, data_loss, phy_loss, val_total_loss, val_phy_loss, val_data_loss = [], [], [], [], [], []
                loss_phy1_list, loss_phy2_list, loss_phy3_list = [], [], []
                loss_data1_list, loss_data2_list, loss_data3_list = [], [], []
                alpha1_list, alpha2_list, alpha3_list = [], [], []
                for epoch in range(self.adam_epochs):
                    print('epoch',epoch)
                    idx = tf.range(len(X_train_data))
                    idx = tf.random.shuffle(idx)
                    train_ind = idx[:-20]
                    val_ind = idx[-20:]
                    X_train_fold, X_val_fold = X_train_data[train_ind,:,:], X_train_data[val_ind,:,:]
                    Y_train_fold, Y_val_fold = Y_train_data[train_ind,:,:], Y_train_data[val_ind,:,:]
                    max_Y_train_data = max_Y_train[train_ind,:,:]
                    max_Y_val_data = max_Y_train[val_ind,:,:]
                    f1_train_fold, f2_train_fold, f3_train_fold = f1_train[train_ind,:,:], f2_train[train_ind,:,:], f3_train[train_ind,:,:]
                    f1_val_fold, f2_val_fold, f3_val_fold = f1_train[val_ind,:,:], f2_train[val_ind,:,:], f3_train[val_ind,:,:]
                    with tf.GradientTape() as tape:
                        output = model(X_train_fold)
                        loss, loss_data1, loss_data2, loss_data3, loss_phy1, loss_phy2, loss_phy3, alpha1, alpha2, alpha3 = self.custom_loss(
                            Y_train_fold, output, max_Y_train_data, f1_train_fold, f2_train_fold, f3_train_fold, alpha)
                    grads = tape.gradient(loss, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    Y_pred_val = model.predict(X_val_fold)
                    loss_val, loss_data1_val, loss_data2_val, loss_data3_val, loss_phy1_val, loss_phy2_val, loss_phy3_val, _, _, _ = self.custom_loss(
                    Y_val_fold, Y_pred_val, max_Y_val_data, f1_val_fold, f2_val_fold, f3_val_fold, alpha)

                    if loss_val < best_val_loss:
                        num_val = epoch
                        model.save(result_file_name + 'best_val_model.h5')
                        best_val_loss = loss_val
                    if loss < best_train_loss:
                        num_train = epoch
                        model.save(result_file_name + 'best_train_model.h5')
                        best_train_loss = loss


                    total_loss.append(loss.numpy())
                    data_loss.append((loss_data1 + loss_data2 + loss_data3).numpy())
                    phy_loss.append((loss_phy1 + loss_phy2 + loss_phy3).numpy())
                    loss_phy1_list.append(loss_phy1.numpy())
                    loss_phy2_list.append(loss_phy2.numpy())
                    loss_phy3_list.append(loss_phy3.numpy())
                    loss_data1_list.append(loss_data1.numpy())
                    loss_data2_list.append(loss_data2.numpy())
                    loss_data3_list.append(loss_data3.numpy())
                    alpha1_list.append(alpha1)
                    alpha2_list.append(alpha2)
                    alpha3_list.append(alpha3)
                    val_total_loss.append(loss_val.numpy())
                    val_phy_loss.append((loss_phy1_val + loss_phy2_val + loss_phy3_val).numpy())
                    val_data_loss.append((loss_data1_val + loss_data2_val + loss_data3_val).numpy())
                    print(f'total_loss: {loss.numpy()}', f'phy_loss: {(loss_phy1 + loss_phy2 + loss_phy3).numpy()}',
                        f'data_loss: {(loss_data1 + loss_data2 + loss_data3).numpy()}')
                    print(f'total_loss_val: {loss_val.numpy()}',
                        f'phy_loss_val: {(loss_phy1_val + loss_phy2_val + loss_phy3_val).numpy()}',
                        f'data_loss_val: {(loss_data1_val + loss_data2_val + loss_data3_val).numpy()}\n')
                    print(f'epoch time:{time.time() - tim} s')
                    tim = time.time()
                    print(f'epoch time:{time.time() - tim} s')
                    tim = time.time()

                duration_t = (time.time() - start_time) / 3600
                print(f'Total training time: {duration_t} hours')

                plt.figure(figsize=(12, 6))
                plt.plot(total_loss, label='Training Loss')
                plt.plot(val_total_loss, label='Validation Loss')
                plt.title('Training and Validation Loss Over Time')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.semilogy()
                plt.legend()
                plt.savefig(os.path.join(result_file_name, 'loss.png'), format='png', dpi=400, bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(12, 6))
                plt.plot(loss_phy1_list, label='phy1')
                plt.plot(loss_phy2_list, label='phy2')
                plt.plot(loss_phy3_list, label='phy3')
                plt.plot(loss_data1_list, label='data1')
                plt.plot(loss_data2_list, label='data2')
                plt.plot(loss_data3_list, label='data3')
                plt.semilogy()
                plt.title('Phy Loss Over Time')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(os.path.join(result_file_name, 'phyloss.png'), format='png', dpi=400, bbox_inches='tight')
                plt.close()



                plt.figure(figsize=(12, 6))
                plt.plot(total_loss, label='Training Loss')
                plt.plot(data_loss, label='Data Loss')
                plt.plot(phy_loss, label='Phy Loss')
                plt.semilogy()
                plt.title('Training and Phy Data Loss Over Time')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(os.path.join(result_file_name, 'loss_phy_data.png'), format='png', dpi=400, bbox_inches='tight')
                plt.close()

                Y_train_pred = model.predict(X_train_data)
                Y_test_pred = model.predict(X_test_data)
                Y_train_pred = Y_train_pred * max_Y_data[train_index]
                Y_test_pred = Y_test_pred * max_Y_data[test_index]
                y_test_orig = Y_test_data * max_Y_data[test_index]
                x_test_orig = X_test_data * max_X_data[test_index]
                y_train_orig = Y_train_data * max_Y_data[train_index]
                x_train_orig = X_train_data * max_X_data[train_index]
                results_psa = self.calculate_statistics(Y_test_data[:, :, 0], Y_test_pred[:, :, 0])
                results_psv = self.calculate_statistics(Y_test_data[:, :, 1], Y_test_pred[:, :, 1])
                results_sd = self.calculate_statistics(Y_test_data[:, :, 2], Y_test_pred[:, :, 2])
                model_trainbest = tf.keras.models.load_model(filepath=result_file_name + 'best_train_model.h5',
                                                             custom_objects={
                                                                             'custom_loss': self.custom_loss})
                Y_train_pred_trainbest = model_trainbest.predict(X_train_data)
                Y_test_pred_trainbest = model_trainbest.predict(X_test_data)
                Y_train_pred_trainbest = Y_train_pred_trainbest * max_Y_data[train_index]
                Y_test_pred_trainbest = Y_test_pred_trainbest * max_Y_data[test_index]
                results_trainbest_psa = self.calculate_statistics(y_test_orig[:, :, 0], Y_test_pred_trainbest[:, :, 0])
                results_trainbest_psv = self.calculate_statistics(y_test_orig[:, :, 1], Y_test_pred_trainbest[:, :, 1])
                results_trainbest_sd = self.calculate_statistics(y_test_orig[:, :, 2], Y_test_pred_trainbest[:, :, 2])
                model_validbest = tf.keras.models.load_model(filepath=result_file_name + 'best_val_model.h5',
                                                             custom_objects={
                                                                             'custom_loss': self.custom_loss})
                Y_train_pred_validbest = model_validbest.predict(X_train_data)
                Y_test_pred_validbest = model_validbest.predict(X_test_data)
                Y_train_pred_validbest = Y_train_pred_validbest * max_Y_data[train_index]
                Y_test_pred_validbest = Y_test_pred_validbest * max_Y_data[test_index]
                results_validbest_psa = self.calculate_statistics(y_test_orig[:, :, 0], Y_test_pred_validbest[:, :, 0])
                results_validbest_psv = self.calculate_statistics(y_test_orig[:, :, 1], Y_test_pred_validbest[:, :, 1])
                results_validbest_sd = self.calculate_statistics(y_test_orig[:, :, 2], Y_test_pred_validbest[:, :, 2])
                results_list = [results_psa, results_psv, results_sd]
                results_trainbest_list = [results_trainbest_psa, results_trainbest_psv, results_trainbest_sd]
                results_validbest_list = [results_validbest_psa, results_validbest_psv, results_validbest_sd]
                spectra = ['PSA', 'PSV', 'SD']

                for sam in range(10):
                    plt.figure()
                    plt.plot(Tn,y_test_orig[sam, :, 0], label='orig')
                    plt.plot(Tn,Y_test_pred_validbest[sam, :, 0], label='valbest')
                    plt.plot(Tn,Y_test_pred_trainbest[sam, :, 0], label='trainbest')
                    plt.semilogx()
                    plt.legend()
                    plt.savefig(os.path.join(result_file_name, f'psa_{sam + 1}.png'), format='png', facecolor='w',
                                dpi=400, bbox_inches='tight')
                    plt.close()
                    plt.figure()
                    plt.plot(Tn,y_test_orig[sam, :, 1], label='orig')
                    plt.plot(Tn,Y_test_pred_validbest[sam, :, 1], label='valbest')
                    plt.plot(Tn,Y_test_pred_trainbest[sam, :, 1], label='trainbest')
                    plt.semilogx()
                    plt.legend()
                    plt.savefig(os.path.join(result_file_name, f'psv_{sam + 1}.png'), format='png', facecolor='w',
                                dpi=400, bbox_inches='tight')
                    plt.close()
                    plt.figure()
                    plt.plot(Tn,y_test_orig[sam, :, 2], label='orig')
                    plt.plot(Tn,Y_test_pred_validbest[sam, :, 2], label='valbest')
                    plt.plot(Tn,Y_test_pred_trainbest[sam, :, 2], label='trainbest')
                    plt.semilogx()
                    plt.legend()
                    plt.savefig(os.path.join(result_file_name, f'sd_{sam + 1}.png'), format='png', facecolor='w',
                                dpi=400, bbox_inches='tight')
                    plt.close()
                for result, result_trainbest, result_validbest, spectrum in zip(results_list, results_trainbest_list,
                                                                                results_validbest_list, spectra):
                    print(f'ave_r2_last_{spectrum}', np.mean(result['r2']), 'RMSE', np.mean(result['rmse']), 'MAE',
                          np.mean(result['mae']))
                    print(f'ave_r2_trainbest_{spectrum}', np.mean(result_trainbest['r2']), 'RMSE',
                          np.mean(result_trainbest['rmse']), 'MAE', np.mean(result_trainbest['mae']))
                    print(f'ave_r2_validbest_{spectrum}', np.mean(result_validbest['r2']), 'RMSE',
                          np.mean(result_validbest['rmse']), 'MAE', np.mean(result_validbest['mae']))
                with open(result_file_name + '均值对比.txt', 'w') as f:
                    for result, result_trainbest, result_validbest, spectrum in zip(results_list,
                                                                                    results_trainbest_list,
                                                                                    results_validbest_list, spectra):
                        f.write(
                            f'ave_r2_last_{spectrum} {np.mean(result["r2"])} RMSE {np.mean(result["rmse"])} MAE {np.mean(result["mae"])}\n')
                        f.write(
                            f'ave_r2_trainbest_{spectrum} {np.mean(result_trainbest["r2"])} RMSE {np.mean(result_trainbest["rmse"])} MAE {np.mean(result_trainbest["mae"])}\n')
                        f.write(
                            f'ave_r2_validbest_{spectrum} {np.mean(result_validbest["r2"])} RMSE {np.mean(result_validbest["rmse"])} MAE {np.mean(result_validbest["mae"])}\n')
                    f.write(f'num_val:{num_val} num_train:{num_train}')
                results_all = {
                    'total_loss': total_loss,
                    'phy_loss': phy_loss,
                    'data_loss': data_loss,
                    'val_total_loss': val_total_loss,
                    'val_phy_loss': val_phy_loss,
                    'val_data_loss': val_data_loss,
                    'x_train_orig': x_train_orig,
                    'y_train_orig': y_train_orig,
                    'x_test_orig': x_test_orig,
                    'y_test_orig': y_test_orig,
                    'Y_test_pred': Y_test_pred,
                    'Y_train_pred': Y_train_pred,
                    'Y_train_pred_validbest': Y_train_pred_validbest,
                    'Y_test_pred_validbest': Y_test_pred_validbest,
                    'learning_rate': 1e-3,
                    'decay': 1e-4,
                    'epoch': self.adam_epochs,
                    'Y_train_pred_trainbest': Y_train_pred_trainbest,
                    'Y_test_pred_trainbest': Y_test_pred_trainbest,
                    'duration_t': duration_t,
                    'alpha1_list': alpha1_list,
                    'alpha2_list': alpha2_list,
                    'alpha3_list': alpha3_list,
                    'loss_phy1_list': loss_phy1_list,
                    'loss_phy2_list': loss_phy2_list,
                    'loss_phy3_list': loss_phy3_list,
                    'loss_data1_list': loss_data1_list,
                    'loss_data2_list': loss_data2_list,
                    'loss_data3_list': loss_data3_list,
                }


                for result, result_trainbest, result_validbest, spectrum in zip(results_list, results_trainbest_list,
                                                                                results_validbest_list, spectra):
                    results_all.update({
                        f'r2_trainbest_{spectrum}': result_trainbest['r2'],
                        f'ave_r2_trainbest_{spectrum}': np.mean(result_trainbest['r2']),
                        f'r2_validbest_{spectrum}': result_validbest['r2'],
                        f'ave_r2_validbest_{spectrum}': np.mean(result_validbest['r2']),
                        f'ave_r2_last_{spectrum}': np.mean(result["r2"]),
                        f'ave_RMSE_last{spectrum}': np.mean(result["rmse"]),
                        f'ave_MAE_last{spectrum}': np.mean(result["mae"]),
                        f'RMSE_trainbest_{spectrum}': result_trainbest['rmse'],
                        f'ave_RMSE_trainbest_{spectrum}': np.mean(result_trainbest['rmse']),
                        f'RMSE_validbest_{spectrum}': result_validbest['rmse'],
                        f'ave_RMSE_validbest_{spectrum}': np.mean(result_validbest['rmse']),
                        f'MAE_trainbest_{spectrum}': result_trainbest['mae'],
                        f'ave_MAE_trainbest_{spectrum}': np.mean(result_trainbest['mae']),
                        f'MAE_validbest_{spectrum}': result_validbest['mae'],
                        f'ave_MAE_validbest_{spectrum}': np.mean(result_validbest['mae']),
                    })


                scipy.io.savemat(result_file_name + 'result_last.mat', results_all)
