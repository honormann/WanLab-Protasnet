import numpy as np
import pandas as pd
from tqdm import tqdm

from CallBack import GetCallback
import tensorflow as tf
from sklearn import metrics

from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score

class Trainer:

    def __init__(self, net=None, dataLoader=None):
        self.net = net
        self.dataLoader = dataLoader

    def CV_params(self, batch_size=None,
                  learning_rate=None,
                  epoch=None,
                  l2=None,
                  droupout=None):

        cv_params = []
        for i in range(len(batch_size)):
            for j in range(len(learning_rate)):
                for k in range(len(epoch)):
                    for l in range(len(l2)):
                        for m in range(len(droupout)):
                            cv_params.append([batch_size[i], learning_rate[j], epoch[k], l2[l], droupout[m]])


        self.cv_params = cv_params


    def CV_search(self):
        print('grid search for best params')
        pbar = tqdm(total=len(self.cv_params), position=0, leave=True)
        opt_loss = np.Inf
        for item in self.cv_params:
            batch, lr, epoch, l2, droupout = item
            self.net.args["batch_size"] = batch
            self.net.args["learning_rate"] = lr
            self.net.args["epoch"] = epoch
            self.net.args["l2"] = l2
            self.net.args["dropout"] = droupout
            self.renew_model()
            val_loss = self.training(verbose=0)
            if val_loss < opt_loss:
                opt_loss = val_loss
                self.best_param = item

            pbar.update(1)

    def train_with_best_param(self):

        print('training with best params')
        batch, lr, epoch, l2, droupout = self.best_param
        self.net.args["batch_size"] = batch
        self.net.args["learning_rate"] = lr
        self.net.args["epoch"] = epoch
        self.net.args["l2"] = l2
        self.net.args["dropout"] = droupout
        self.renew_model()
        self.training()

    def renew_model(self):
        self.net.build_model()
        if self.dataLoader.datatype == 'bulk':
            self.net.compile_model(type='cox')
        else:
            self.net.compile_model(type='class')


    def training(self, verbose='auto'):

        callbacks = GetCallback(self.net.args)

        if self.dataLoader.datatype == 'bulk':
            y_train = tf.concat([self.dataLoader.yevent_train, self.dataLoader.ytime_train], axis=1)
            y_valid = tf.concat([self.dataLoader.yevent_val, self.dataLoader.ytime_val], axis=1)
            history = self.net.protasnet.fit(x=self.dataLoader.x_train, y=y_train, epochs=self.net.args["epoch"],
                              validation_data=(self.dataLoader.x_val, y_valid),
                              callbacks=callbacks,
                              batch_size=self.net.args["batch_size"],
                              verbose=verbose)
        else:
            history = self.net.protasnet.fit(x=self.dataLoader.x_train, y=self.dataLoader.y_train,
                                            epochs=self.net.args["epoch"],
                                            validation_data=(self.dataLoader.x_val, self.dataLoader.y_val), callbacks=callbacks,
                                            batch_size=self.net.args["batch_size"],
                                            class_weight=self.dataLoader.class_weight,
                                            verbose=verbose)

        return history.history['val_loss'][-1]

    def perform(self, y_true, y_pred):
        pred = (y_pred.numpy() >= 0.5) * 1
        acc = metrics.accuracy_score(y_true, pred)
        auc = metrics.roc_auc_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, pred)
        recall = metrics.accuracy_score(y_true, pred)

        return acc, auc, f1_score, recall

    def evaluate(self):

        ans = self.net.protasnet(self.dataLoader.x_test)
        pred = tf.reduce_mean(ans, axis=0)

        if self.dataLoader.datatype == 'bulk':
            pred = tf.reshape(pred, shape=(-1,))
            pred = pred.numpy()

            y = pd.DataFrame({"os": self.dataLoader.yevent_train.numpy().reshape(-1),
                              "time": self.dataLoader.ytime_train.numpy().reshape(-1)}).to_numpy()
            aux = [(e1 == 1, e2) for e1, e2 in y]
            new_data_y = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
            y = pd.DataFrame({"os": self.dataLoader.yevent_test.numpy().reshape(-1),
                              "time": self.dataLoader.ytime_test.numpy().reshape(-1)}).to_numpy()
            aux = [(e1 == 1, e2) for e1, e2 in y]
            new_data_test = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

            try:
                cIndex = concordance_index_censored([item == 1 for item in self.dataLoader.yevent_test.numpy().reshape(-1)],
                                                    self.dataLoader.ytime_test.numpy().reshape(-1), pred, tied_tol=1e-08)
            except:
                cIndex = [0]

            try:
                cIndex_ipcw = concordance_index_ipcw(new_data_y, new_data_test, pred)
            except:
                cIndex_ipcw = [0]

            try:
                _, brier = brier_score(new_data_y, new_data_test, pred, times=np.median(self.dataLoader.X["phe_os_time"]))
            except:
                brier = [0]

            self.cIndex, self.cIndex_ipcw, self.brier_score = cIndex[0], cIndex_ipcw[0],brier[0]


        else:
            self.acc, self.auc, self.f1_score, self.recall = self.perform(self.dataLoader.y_test, pred)

