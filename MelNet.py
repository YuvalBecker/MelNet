# Melnet -> Replicating FAIR article.  using tensorflow :https://arxiv.org/pdf/1906.01083.pdf
# At the moment produces unconditional generation :
from tensorflow.keras.layers import LSTM, Conv2D, Input, Concatenate
import tensorflow.keras.backend as K
from making_mel_Spec import create_data_train, return_to_audio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Input, Lambda
import tensorflow_probability as tfp
import gc
tfd = tfp.distributions


def generating_from_disribution_new(tensor_out):
    # Sampling from a gaussian mixture model:
    miut1 = tensor_out[:, 1, :, :, :]
    sigmat1 = tensor_out[:, 2, :, :, :] * 0.5
    alphat1 = tensor_out[:, 0, :, :, :]

    dim_f = np.size(miut1, 1)
    dim_t = np.size(miut1, 2)
    batch_size = np.size(miut1, 0)
    num_mixture = int(np.size(miut1, 3))

    # data dimentions : ( Batch_size , dim_f , dim_t,3,k )
    # Apply transformation as in (3) (4) (5) to gaussian model:
    output_tot = np.zeros((dim_f, dim_t))
    output_tot_batch_format = np.zeros((1, dim_f, dim_t))

    for batch_num in range(batch_size):
        alpha_batch = alphat1[batch_num]
        miu_batch = miut1[batch_num]
        sigma_batch = sigmat1[batch_num]
        sum_alpha_batch = np.sum(alpha_batch, axis=2)
        for l in range(num_mixture):
            alpha_batch[:, :, l] = alpha_batch[:, :, l] / sum_alpha_batch
        out = np.zeros((dim_f, dim_t))
        for i in range(dim_f):
            for j in range(dim_t):
                ind = np.random.choice(np.arange(0, num_mixture),
                                       p=np.ravel([alpha_batch[i, j, :]]))
                out[i, j] = np.random.normal(miu_batch[i, j, ind],
                                             0.221 * sigma_batch[i, j, ind])
        output_tot = np.append(output_tot, out, axis=1)
        output_tot_batch_format = np.append(output_tot_batch_format, np.expand_dims(out, axis=0), axis=0)
    return output_tot, output_tot_batch_format


def generating(Input_In):
    count1 = 1
    input_generator = Input_In[range(count1 * batch_size1, count1 * batch_size1 + batch_size1), :, :]
    input_generator = input_generator - np.min(input_generator)
    input_generator = input_generator / np.max(input_generator)

    col = 20
    col1 = 20
    input_data2 = 0 * np.random.random((batch_size1, dim_f1, 1 * dim_t1)) * np.mean(input_generator)
    input_data2[:, :, 0:col] = input_generator[0:batch_size1, :, 0:col]
    out_tensor = model.predict(input_data2)
    params = Mel_net.sampling_gauss(out_tensor)
    output_from_Generation, output_from_Generation_batch = generating_from_disribution_new(out_tensor)
    # sanity check before generation:(Easy generation at first)
    output_from_Generation_batch = params.sample()
    plt.figure(31)
    plt.imshow(output_from_Generation_batch[11, :])
    plt.figure(32)
    plt.imshow((output_from_Generation[:, 50:]))

    # Generation :PIXEL BY PIXEL [it takes lots of time]
    input_show = np.zeros((np.shape(input_data2)))
    for ik in range(dim_t1 - col):
        for jk in range(0, dim_f1, 1):
            input_net = input_data2[:, :, 0:dim_t1]
            out_tensor = model.predict(input_net)
            plt.close()
            output_from_Generation, output_from_Generation_batch = generating_from_disribution_new(out_tensor)
            # params = sampling_gauss(out_tensor)
            # output_from_Generation_batch = params.sample()

            temp_amplitude = output_from_Generation_batch
            temp_amplitude = output_from_Generation_batch[1:, :, :]
            input_data2[:, jk, col] = temp_amplitude[:, jk, col]
            input_show[:, jk, col] = temp_amplitude[:, jk, col]
            plt.figure(2)
            plt.imshow(input_data2[0, :])
            plt.pause(0.1)
        col = col + 1

    # Gathering all the batches to 1 long sequence--> Viewing purposes
    new = np.zeros((dim_f1, dim_t1))
    for ii in range(batch_size1):
        new = np.append(new, input_data2[ii, :, col1:], axis=1)
    OUT = new[:, 50:]
    plt.figure(4)
    plt.imshow(OUT)

    ## Returning back to audio:
    rate = 24000
    FS = rate
    return return_to_audio(FS, OUT[:, 2:])


def train_data(input_data=None, batch_size=None, num_epochs=200, num_steps_over_batch=4, mel_model=None,path_save='.'):
    # very simple train function:
    count = 0
    count2 = 0
    # Trainning the model:
    for n in range(num_epochs):
        gc.collect()
        print('iter num:' + str(n))
        index1 = np.mod(count, np.size(input_data, 0))
        if index1 + batch_size - 1 >= np.size(input_data, 0):
            count = 0
        input_data_batch = input_data[range(count, count + batch_size), :, :]
        # adding noise
        input_data_batch = input_data_batch - np.min(input_data_batch)
        input_data_batch = input_data_batch / np.max(input_data_batch)
        input_data_batch_noise = input_data_batch + 0 * np.random.random(
            (np.shape(input_data_batch))) * np.mean(
            input_data_batch) * 0.5
        input_data_batch_noise[:, :, dim_t1 - 1] = np.random.random(
            (np.shape(input_data_batch[:, :, dim_t1 - 1]))) * np.mean(input_data_batch) * 1
        count = count + batch_size
        for i in range(num_steps_over_batch):
            result = mel_model.train_on_batch(input_data_batch_noise, input_data_batch_noise)
            print(result)
            count2 = count2 + 1
        model.save(path_save+'model_epoch'+str(n)+'.h5')
    return


class Melnet:
    def __init__(self,dim_f, dim_t, hidden_layer=3,mixture_num=10):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.hidden_layer = hidden_layer
        self.mixture_num = mixture_num
    # functions related to running the tranning process - arrange the data

    def prepare_inputs_net(self,spectrogram):
        # an option is to padded with zeros instead rotating:
        # delay input Freq -> for input to FrequencyDelayedStack
        freq_input_matrix = tf.roll(spectrogram, shift=1, axis=1)
        # delay input Time -> for input to TimeDelayedStack
        time_input_matrix = tf.roll(spectrogram, shift=1, axis=2)
        # delay input Time -> for input to TimeDelayedStack -> for frequency rnn
        freq_timeStack_input = tf.transpose(time_input_matrix, [0, 2, 1])

        return freq_input_matrix, time_input_matrix, freq_timeStack_input

    # Main MelNet function:
    def MelNET(self,is_trainning=True):
        # MElNET Architecture :
        # Trainning parameters:
        spectrogram = Input(shape=(self.dim_f, self.dim_t))

        freq_input_matrix, Time_input_matrix, Freq_timeStack_input= self.prepare_inputs_net(spectrogram)

        # Running the TimeDelayedStack first in order to insert it to FrequencyDelayedStack
        concat_output = self._TimeDelayedStack(Time_input_matrix, Freq_timeStack_input, self.dim_f,
                                         self.dim_t, self.hidden_layer)

        concat_output = tf.keras.layers.BatchNormalization()(concat_output, training=is_trainning)

        # Define the matrics matmul weights variable:
        matwh1 = tf.Variable(shape=[3 * self.hidden_layer, self.hidden_layer], dtype=tf.float32,
                             initial_value=tf.ones(shape=[3 * self.hidden_layer, self.hidden_layer]))
        tot_matrix = tf.transpose(tf.matmul(concat_output, matwh1), [0, 3, 2, 1])
        ##### Calculating over all hidden layers:

        # arranging the residual part in time to match the size of hidden layers:
        t_concat = tf.expand_dims(tf.transpose(Time_input_matrix, [0, 2, 1]), axis=1)

        # Residual connections for lstm outputs :
        for i in range(0, self.hidden_layer - 1, 1):
            t_concat = Concatenate(axis=1)([t_concat, tf.expand_dims(tot_matrix[:, i, :, :], axis=1)])
        # creating input for frequencyDelayStack for time residual:
        time_output = tot_matrix + t_concat
        # calculating  frequencyDelayStack :
        out_freq = self._FrequencyDelayedStack(freq_input_matrix, time_output, self.dim_f, self.dim_t, self.hidden_layer)
        out_freq = tf.keras.layers.BatchNormalization()(out_freq, training=is_trainning)

        # Calculating mixture gaussian parameters :
        mu = Conv2D(filters=self.mixture_num, kernel_size=(1, 1), activation='relu', padding='same')(
            tf.expand_dims(out_freq, axis=3))
        mu = Conv2D(filters=self.mixture_num, kernel_size=(1, 1), activation='relu', padding='same')(
            mu)
        mu = tf.keras.layers.BatchNormalization()(mu, training=is_trainning)
        sigma = Conv2D(filters=self.mixture_num, kernel_size=(1, 1), activation='relu', padding='same')(
            tf.expand_dims(out_freq, axis=3))
        sigma = Conv2D(filters=self.mixture_num, kernel_size=(1, 1), activation='relu', padding='same')(sigma)

        sigma = tf.keras.layers.BatchNormalization()(sigma, training=is_trainning)
        sigma = K.exp(sigma)
        alpha = Conv2D(filters=self.mixture_num, kernel_size=(1, 1), activation='relu', padding='same')(
            tf.expand_dims(out_freq, axis=3))
        alpha = Conv2D(filters=self.mixture_num, kernel_size=(1, 1), activation='relu', padding='same')(alpha)

        alpha = tf.keras.layers.BatchNormalization()(alpha, training=is_trainning)
        alpha = K.exp(alpha)

        # mixture sum to one :
        alpha = tf.expand_dims(alpha[:, :, :, 0] / K.sum(alpha, axis=3), axis=3)

        for i in range(1, self.mixture_num, 1):
            alpha = tf.concat([alpha, tf.expand_dims(alpha[:, :, :, i] / K.sum((alpha), axis=3), axis=3)], axis=3)

        tensor_out = Concatenate(axis=0)(
            [tf.expand_dims(alpha, axis=0), tf.expand_dims(mu, axis=0), tf.expand_dims(sigma, axis=0)])
        tensor_out = tf.transpose(tensor_out, [1, 0, 2, 3, 4])

        model = Model(inputs=spectrogram, outputs=tensor_out)

        return model

    # functions related to graph creation
    def _FrequencyDelayedStack(self,freq_input_matrix, time_output, dim_f, dim_t, hidden_layer):
        # inputs :
        # freq_input_matrix : martix shifted in the frequency domain.
        # time_output: output from the time delay stack
        # dims : input dimention
        Time_input = tf.transpose(time_output, [0, 1, 3, 2])  # tf.expand_dims(time_output, axis=0)
        freq_vec_input = tf.expand_dims(freq_input_matrix[:, 0:dim_f, 0], axis=2)
        # first :
        lstm_freq = LSTM(1, input_shape=(None, dim_f, 1), return_state=True, return_sequences=True,
                         recurrent_dropout=0.2, dropout=0.2,
                         name='freq_Freq')
        out_freq_lstm = lstm_freq(
            freq_vec_input)  # run over time samples
        tot_hidden = out_freq_lstm[0]
        for i in range(1, dim_t):
            out2 = tf.expand_dims(freq_input_matrix[:, 0:dim_f, i], axis=2)
            out_freq_lstm = lstm_freq(
                out2)  # run over time samples
            temp_out_lstm = out_freq_lstm[0]
            # Each layer match to the specific layer in TimeDelayStack
            for j in range(hidden_layer):
                W = tf.Variable(shape=[32, 1], dtype=tf.float32, initial_value=np.ones((32, 1)) * 0.1)
                input11 = W * temp_out_lstm + tf.expand_dims(Time_input[:, j, 0:dim_f, i], axis=2)
                temp_out_lstm = lstm_freq(input11)[0] + temp_out_lstm
            tot_hidden = tf.concat([tot_hidden, temp_out_lstm], axis=2)
        # output = tot_hidden + freq_input_matrix
        output = tot_hidden
        return output

    def _TimeDelayedStack(self,Time_input_matrix, Freq_timeStack_input, dim_f, dim_t, hidden_layer):
        # inputs :
        # Time_input_matrix -> containing spectorgram shaped for time sequence
        # freq_input_mat-> spectrogram shaped for freq sequence
        # dim_f -> dimension over the freq axis
        # dim_t -> dimenstion over the time axis
        # hidden_layer -> number of layers of the lstm
        # define LSTM operation -> shared weights:
        # summing over all time shared weights outputs of the lstm :
        # RNN 1 -> Time axis :
        time_vec_input = tf.expand_dims(Time_input_matrix[:, 0, 0:dim_t], axis=1)

        lstm_moudule_time = LSTM(hidden_layer, input_shape=(None, dim_t, 1), return_state=True,
                                 dropout=0.2, recurrent_dropout=0.2,
                                 return_sequences=True, name='time_time')
        out_time1_lstm = lstm_moudule_time(tf.transpose(time_vec_input, (0, 2, 1)))  # run over time samples
        tot_hidden = tf.expand_dims(out_time1_lstm[0], axis=2)
        for i in range(1, dim_f):
            time_vec_input = tf.expand_dims(Time_input_matrix[:, i, 0:dim_t], axis=1)
            out_time1_lstm = lstm_moudule_time(tf.transpose(time_vec_input, (0, 2, 1)))
            tot_hidden = Concatenate(axis=2)([tot_hidden, tf.expand_dims(out_time1_lstm[0], axis=2)])

        # RNN 2 -> inverted frequency axis
        # summing over all frequency reveresed ->  shared weights outputs of the lstm :

        freq_vec_input_flipped = tf.expand_dims(Freq_timeStack_input[:, 0, dim_f:None:-1], axis=1)

        lstm_freq_moudule_backward = LSTM(hidden_layer, input_shape=(None, dim_f, 1), return_state=True,
                                          recurrent_dropout=0.2, dropout=0.2,
                                          return_sequences=True, name='freq_revers_time')
        out_freq_lstm_flipped = lstm_freq_moudule_backward(
            tf.transpose(freq_vec_input_flipped, (0, 2, 1)))  # run over time samples
        tot_hidden_freq_flipped = tf.expand_dims(out_freq_lstm_flipped[0], axis=2)
        for i in range(1, dim_t):
            freq_vec_input_flipped = tf.expand_dims(Freq_timeStack_input[:, i, dim_f:None:-1], axis=1)
            out_lstm_freq = lstm_freq_moudule_backward(tf.transpose(freq_vec_input_flipped, (0, 2, 1)))
            tot_hidden_freq_flipped = tf.concat([tot_hidden_freq_flipped, tf.expand_dims(out_lstm_freq[0], axis=2)],
                                                axis=2)

        # RNN 3 -> frequency axis
        # summing over all frequency shared weights outputs of the lstm :
        freq_vec_input = tf.expand_dims(Freq_timeStack_input[:, 0, 0:dim_f], axis=1)

        lstm_freq_moudule_forward = LSTM(hidden_layer, input_shape=(None, dim_f, 1), return_state=True,
                                         recurrent_dropout=0.2, dropout=0.2,
                                         return_sequences=True, name='freq_time')
        out_freq_lstm = lstm_freq_moudule_forward(tf.transpose(freq_vec_input, (0, 2, 1)))  # run over time samples
        tot_hidden_freq = tf.expand_dims(out_freq_lstm[0], axis=2)
        for i in range(1, dim_t):
            freq_vec_input = tf.expand_dims(Freq_timeStack_input[:, i, 0:dim_f], axis=1)

            out_lstm_freq = lstm_freq_moudule_forward(tf.transpose(freq_vec_input, (0, 2, 1)))
            tot_hidden_freq = tf.concat([tot_hidden_freq, tf.expand_dims(out_lstm_freq[0], axis=2)], axis=2)
        # Concatination of the 3 LSTMs outputs:
        tot_hidden = tf.transpose(tot_hidden, [0, 2, 1, 3])
        CONCAT_3_RNN_OUPUT = (tf.concat([tot_hidden, tot_hidden_freq_flipped, tot_hidden_freq], axis=3))

        return CONCAT_3_RNN_OUPUT


    def gaussian_mixture_loss(self,label_spect, tensor_out):
        # Dimentions: (batch_size,dim_f1,dim_t1,num of mixture)
        mu = tensor_out[:, 1, :]
        sigma = tensor_out[:, 2, :]
        alpha = tensor_out[:, 0, :]

        gm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=alpha),
            components_distribution=tfd.Normal(
                loc=mu,
                scale=sigma))
        log_loss = -tf.reduce_sum(gm.log_prob(tf.squeeze(label_spect)))
        return log_loss


    def sampling_gauss(self,tensor_out):
        # Dimentions: (batch_size,dim_f1,dim_t1,num of mixture)
        mu = tensor_out[:, 1, :]
        sigma = tensor_out[:, 2, :] * 0.5
        alpha = tensor_out[:, 0, :]

        gm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=alpha),
            components_distribution=tfd.Normal(
                loc=mu,
                scale=sigma))
        return gm


if __name__ == "__main__":
    # Creating the graph of MELNET model:
    dict_inputs = {'dim_f1': 32, 'dim_t1': 50, 'hidden_layer1': 10}
    Mel_net = Melnet(*dict_inputs)
    model = Mel_net.MelNET(is_trainning = True)

    filepath = r""
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='min')
    logdir = r".\logs"
    tensorboard_callback = TensorBoard(log_dir=logdir)

    callbacks_list = [tensorboard_callback]

    model.compile('RMSprop', lr=5e-4, loss=Mel_net.gaussian_mixture_loss, metrics=[Mel_net.gaussian_mixture_loss],
                  callbacks=callbacks_list)

    model.load_weights("path_to_my_model.h5")
    # Trainning loop :
    batch_size1 = 25
    num_epochs1 = 500
    num_steps_over_batch1 = 4  # how many repetition for gradient decent over the current batch
    #  Folder containning the data (wav files)
    folder_paths = r"Path of your wav files"
    lr = 5e-4
    Input_In = create_data_train(folder_paths)

    dim_f1 = np.size(Input_In, 1)
    dim_t1 = np.size(Input_In, 2)
    train_data(input_data=Input_In, batch_size=batch_size1, num_epochs=int(np.size(Input_In, axis=0) / batch_size1),
               num_steps_over_batch=num_steps_over_batch1, model=model)
