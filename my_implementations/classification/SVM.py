import numpy as np


class LinearSVM():

    def __init__(self, etha=0.01, alpha=0.1, n_iter=200):
        self.etha = etha
        self.alpha = alpha
        self.n_iter = n_iter
        self.w = None

    def fit(self, X_train, y_train):
        X_train = self.__get_addition_feature(X_train)
        # подразумевается, что решается задача бинирной классификации
        label_0, label_1 = np.unique(y_train)
        y_train[y_train == label_0] = -1
        y_train[y_train == label_1] = 1

        self.w = np.random.normal(loc=0, scale=0.05, size=X_train.shape[1])

        for _ in range(self.n_iter):
            for idx in range(X_train.shape[0]):
                margin = y_train[idx] * np.dot(self.w, X_train[idx])

                if margin >= 1:
                    gradient = self.alpha * self.w / self.n_iter # Зачем делить на n_iter?
                    self.w -= self.etha * gradient
                else:
                    gradient = self.alpha * self.w / self.n_iter - y_train[idx] * X_train[idx]
                    self.w -= self.etha * gradient

    def predict(self, X_test):
        X_test = self.__get_addition_feature(X_test)
        y_pred = []

        for idx in range(X_test.shape[0]):
            sign = np.sign(np.dot(self.w, X_test[idx]))
            prediction = 1 if sign >= 0 else 0
            y_pred.append(prediction)

        return np.array(y_pred)

    def __get_addition_feature(self, X):
        return np.hstack((X, np.ones((X.shape[0], 1), dtype=int)))

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)



# def add_bias_feature(a):
#     a_extended = np.zeros((a.shape[0],a.shape[1]+1))
#     a_extended[:,:-1] = a
#     a_extended[:,-1] = int(1)  
#     return a_extended

# class CustomSVM(object):

#     def __init__(self, etha=0.01, alpha=0.1, epochs=200):
#         self._epochs = epochs
#         self._etha = etha
#         self._alpha = alpha
#         self._w = None
#         self.history_w = []
#         self.train_errors = None
#         self.val_errors = None
#         self.train_loss = None
#         self.val_loss = None

#     def fit(self, X_train, Y_train, X_val, Y_val, verbose=False): #arrays: X; Y =-1,1

#         if len(set(Y_train)) != 2 or len(set(Y_val)) != 2:
#             raise ValueError("Number of classes in Y is not equal 2!")

#         X_train = add_bias_feature(X_train)
#         X_val = add_bias_feature(X_val)
#         self._w = np.random.normal(loc=0, scale=0.05, size=X_train.shape[1])
#         self.history_w.append(self._w)
#         train_errors = []
#         val_errors = []
#         train_loss_epoch = []
#         val_loss_epoch = []

#         for epoch in range(self._epochs): 
#             tr_err = 0
#             val_err = 0
#             tr_loss = 0
#             val_loss = 0
#             for i,x in enumerate(X_train):
#                 margin = Y_train[i]*np.dot(self._w,X_train[i])
#                 if margin >= 1: # классифицируем верно
#                     self._w = self._w - self._etha*self._alpha*self._w/self._epochs
#                     tr_loss += self.soft_margin_loss(X_train[i],Y_train[i])
#                 else: # классифицируем неверно или попадаем на полосу разделения при 0<m<1
#                     self._w = self._w +\
#                     self._etha*(Y_train[i]*X_train[i] - self._alpha*self._w/self._epochs)
#                     tr_err += 1
#                     tr_loss += self.soft_margin_loss(X_train[i],Y_train[i])
#                 self.history_w.append(self._w)
#             for i,x in enumerate(X_val):
#                 val_loss += self.soft_margin_loss(X_val[i], Y_val[i])
#                 val_err += (Y_val[i]*np.dot(self._w,X_val[i])<1).astype(int)
#             if verbose:
#                 print('epoch {}. Errors={}. Mean Hinge_loss={}'\
#                       .format(epoch,err,loss))
#             train_errors.append(tr_err)
#             val_errors.append(val_err)
#             train_loss_epoch.append(tr_loss)
#             val_loss_epoch.append(val_loss)
#         self.history_w = np.array(self.history_w)    
#         self.train_errors = np.array(train_errors)
#         self.val_errors = np.array(val_errors)
#         self.train_loss = np.array(train_loss_epoch)
#         self.val_loss = np.array(val_loss_epoch)                    

#     def predict(self, X:np.array) -> np.array:
#         y_pred = []
#         X_extended = add_bias_feature(X)
#         for i in range(len(X_extended)):
#             y_pred.append(np.sign(np.dot(self._w,X_extended[i])))
#         return np.array(y_pred)         

#     def hinge_loss(self, x, y):
#         return max(0,1 - y*np.dot(x, self._w))

#     def soft_margin_loss(self, x, y):
#         return self.hinge_loss(x,y)+self._alpha*np.dot(self._w, self._w)