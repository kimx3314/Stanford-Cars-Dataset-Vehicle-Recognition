# altered by Sean Sungil Kim                  < https://github.com/kimx3314 >
# reference:                                  < https://github.com/surmenok/keras_lr_finder >
# used for learning rate tuning



import keras        
import keras.backend as K
from keras.callbacks import LambdaCallback
import math
import matplotlib.pyplot as plt



class lr_finder:
    # given that the input model is a keras/tensorflow model
    # plots the loss vs. learning rate (exponentially increasing)
    # helps choosing the optimal learning rate
    
    def __init__(self, model):
        
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9
        
        
        
    def on_batch_end(self, batch, logs):
        
        # recording the learning rate and the loss
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
        loss = logs['loss']
        self.losses.append(loss)

        # checking if the loss got too large or NaN
        if math.isnan(loss) or loss > self.best_loss * 4:
            self.model.stop_training = True
            return
        if loss < self.best_loss:
            self.best_loss = loss

        # increasing the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)
        
        
        
    def find(self, x_train, y_train, start_lr, end_lr, batch_size = 32, epochs = 1):
        
        num_batches = epochs * x_train.shape[0] / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))

        # saving weights
        self.model.save_weights('tmp.h5')

        # original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)
        
        callback = LambdaCallback(on_batch_end = lambda batch, logs: self.on_batch_end(batch, logs))
        self.model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, callbacks = [callback])

        # restoring the weights and the original learning rate
        self.model.load_weights('tmp.h5')
        K.set_value(self.model.optimizer.lr, original_lr)
        
        
        
    def find_generator(self, generator, start_lr, end_lr, epochs = 1, steps_per_epoch = None, **kw_fit):
        
            if steps_per_epoch is None:
                try:
                    steps_per_epoch = len(generator)
                except (ValueError, NotImplementedError) as e:
                    raise e('`steps_per_epoch=None` is only valid for a generator based on the '
                            '`keras.utils.Sequence` class. Please specify `steps_per_epoch` '
                            'or use the `keras.utils.Sequence` class.')
            self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(steps_per_epoch))

            # saving weights
            self.model.save_weights('tmp.h5')

            # original learning rate
            original_lr = K.get_value(self.model.optimizer.lr)

            # initial learning rate
            K.set_value(self.model.optimizer.lr, start_lr)

            callback = LambdaCallback(on_batch_end = lambda batch, logs: self.on_batch_end(batch, logs))
            self.model.fit_generator(generator = generator, epochs = epochs, steps_per_epoch = steps_per_epoch,\
                                     callbacks = [callback], **kw_fit)

            # restoring the weights and the original learning rate
            self.model.load_weights('tmp.h5')
            K.set_value(self.model.optimizer.lr, original_lr)
    
    
    
    def plot_loss(self, n_skip_beginning = 10, n_skip_end = 5):
        
        plt.title('Log Learning Rate vs. Loss', y = 1.02)
        plt.ylabel("Loss"), plt.xlabel("Log Learning Rate")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale('log'), plt.grid(True, which = "both")
        plt.show()
        
        
        
    def plot_loss_change(self, sma = 1, n_skip_beginning = 10, n_skip_end = 5):

        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivative = (self.losses[i] - self.losses[i - sma]) / sma
            derivatives.append(derivative)

        plt.title('Log Learning Rate vs. Rate of Loss Change', y = 1.02)
        plt.ylabel("Rate of Loss Change"), plt.xlabel("Log Learning Rate")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], derivatives[n_skip_beginning:-n_skip_end])
        plt.xscale('log'), plt.grid(True, which = "both")
        plt.show()



