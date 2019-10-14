# Melnet -> Replicating FAIR article.  using tensorflow :https://arxiv.org/pdf/1906.01083.pdf
# At the moment produces unconditional generation :
from keras.layers import LSTM, Conv2D
import keras.backend as K
from making_mel_Spec import create_data_train, return_to_audio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import tensorflow_probability as tfp
import gc

tfd = tfp.distributions


# functions related to running the tranning process - arrange the data
def generating_from_disribution_new(miut1, sigmat1, alphat1):
    # Sampling from a gaussian mixture model:
    dim_f = np.size(miut1, 1)
    dim_t = np.size(miut1, 2)
    batch_size = np.size(miut1, 0)
    num_mixture = int(np.size(miut1, 3))
    # data dimentions : ( Batch_size , dim_f , dim_t,3,k )
    # Apply transformation as in (3) (4) (5) to gaussian model:
    output_tot = np.zeros((dim_f, dim_t))
    output_tot_batch_format = np.zeros((1, dim_f, dim_t))

    for batch_num in range(batch_size):
        sum_mixture = 0
        alpha_batch = alphat1[batch_num]
        miu_batch = miut1[batch_num]
        sigma_batch = sigmat1[batch_num]
        sum_alpha_batch = np.sum(alpha_batch, axis=2)
        for l in range(num_mixture):
            alpha_batch[:, :, l] = alpha_batch[:, :, l] / sum_alpha_batch
        out = np.zeros((dim_f, dim_t))
        if np.sum(1) > 0:
            for i in range(dim_f):
                for j in range(dim_t):
                    ind = np.random.choice(np.arange(0, num_mixture),
                                           p=np.ravel([alpha_batch[i, j, :]]))
                    out[i, j] = np.random.normal(miu_batch[i, j, ind],
                                                 sigma_batch[i, j, ind])
        output_tot = np.append(output_tot, out, axis=1)
        output_tot_batch_format = np.append(output_tot_batch_format, np.expand_dims(out, axis=0), axis=0)
    return output_tot, output_tot_batch_format


def train_data(input_data=None, batch_size=None, num_epochs=200, num_steps_over_batch=4, train_op=None,
               spectrogram=None, labels_spect=None, loss1=None, session=None, learning_rate1=None, lr=1e-4):
    # very simple train function:
    count = 0
    count2 = 0
    # Trainning the model:
    for n in range(num_epochs):
        if np.mod(n, 2) == 0:  # Saving model after 20 iterations:
            saver = tf.train.Saver()
            saver.save(session, r'my-modelneww2' + str(n) + '.ckpt')

        print('iter num:' + str(n))
        index1 = np.mod(count, np.size(input_data, 0))
        if index1 + batch_size - 1 >= np.size(input_data, 0):
            count = 0
        input_data_batch = input_data[range(count, count + batch_size), :, :]
        # adding noise
        input_data_batch_noise = input_data_batch + 0.115 * np.random.random(
            (np.shape(input_data_batch))) * np.mean(
            input_data_batch) * 0.5
        input_data_batch_noise[:, :, dim_t1 - 1] = np.random.random(
            (np.shape(input_data_batch[:, :, dim_t1 - 1]))) * np.mean(input_data_batch) * 2
        count = count + batch_size
        for i in range(num_steps_over_batch):
            result = session.run([train_op, loss1],
                                 feed_dict={spectrogram: input_data_batch_noise, labels_spect: input_data_batch,
                                            learning_rate1: lr})
            print('loss: ' + str(result[1]))
            count2 = count2 + 1
    return


# functions related to graph creation
def FrequencyDelayedStack(freq_input_matrix, time_output, dim_f, dim_t, hidden_layer):
    # inputs :
    # freq_input_matrix : martix shifted in the frequency domain.
    # time_output: output from the TImedelay stack
    # dims : input dimention
    Time_input = time_output  # tf.expand_dims(time_output, axis=0)
    freq_vec_input = tf.expand_dims(freq_input_matrix[:, 0:dim_f, 0], axis=2)
    # first :
    with tf.variable_scope('freq1_lstm') as scope:
        lstm_freq_moudule = LSTM(1, input_shape=(None, dim_f, 1), return_state=True, return_sequences=True,
                                 recurrent_dropout=0.2, dropout=0.2,
                                 name='freq_Freq')
        out_freq_lstm = lstm_freq_moudule(
            freq_vec_input)  # run over time samples
        tot_hidden = out_freq_lstm[0]
        scope.reuse_variables()  # the variables will be reused.
        for i in range(1, dim_t):
            out2 = tf.expand_dims(freq_input_matrix[:, 0:dim_f, i], axis=2)
            out_freq_lstm = lstm_freq_moudule(
                out2)  # run over time samples
            temp_out_lstm = out_freq_lstm[0]

            # Each layer match to the specific layer in TimeDelayStack
            for j in range(hidden_layer):
                input11 = temp_out_lstm + tf.expand_dims(Time_input[:, j, 0:dim_f, i], axis=2)

                temp_out_lstm = lstm_freq_moudule(input11)[0] + temp_out_lstm
            tot_hidden = tf.concat([tot_hidden, temp_out_lstm], axis=2)
            scope.reuse_variables()  # the variables will be reused.
    output = tot_hidden
    return output


def TimeDelayedStack(Time_input_matrix, Freq_timeStack_input, dim_f, dim_t, hidden_layer):
    # inputs :
    # Time_input_matrix -> containing spectorgram shaped for time sequence
    # freq_input_mat-> spectrogram shaped for freq sequence
    # dim_f -> dimension over the freq axis
    # dim_t -> dimenstion over the time axis
    # hidden_layer -> number of layers of the lstm
    # define LSTM operation -> shared weights:
    # summing over all time shared weights outputs of the lstm :
    # RNN 1 -> Time axis :
    with tf.variable_scope('Time_lstm') as scope:
        time_vec_input = tf.expand_dims(Time_input_matrix[:, 0, 0:dim_t], axis=1)

        lstm_moudule_time = LSTM(hidden_layer, input_shape=(None, dim_t, 1), return_state=True,
                                 dropout=0.2, recurrent_dropout=0.2,
                                 return_sequences=True, name='time_time')
        out_time1_lstm = lstm_moudule_time(tf.transpose(time_vec_input, (0, 2, 1)))  # run over time samples
        tot_hidden = tf.expand_dims(out_time1_lstm[0], axis=2)
        scope.reuse_variables()  # the variables will be reused.
        for i in range(1, dim_f):
            time_vec_input = tf.expand_dims(Time_input_matrix[:, i, 0:dim_t], axis=1)

            out_time1_lstm = lstm_moudule_time(tf.transpose(time_vec_input, (0, 2, 1)))
            tot_hidden = tf.concat([tot_hidden, tf.expand_dims(out_time1_lstm[0], axis=2)], axis=2)
            scope.reuse_variables()  # the variables will be reused.
        tot_hidden = tf.math.cumsum(tot_hidden, axis=2)

    # RNN 2 -> inverted frequency axis
    # summing over all frequency reveresed ->  shared weights outputs of the lstm :
    Freq_timeStack_input_fliped = tf.image.flip_left_right(Freq_timeStack_input)
    with tf.variable_scope('freq_lstm_reveres') as scope:
        freq_vec_input_flipped = tf.expand_dims(Freq_timeStack_input_fliped[:, 0, 0:dim_f], axis=1)

        lstm_freq_moudule_backward = LSTM(hidden_layer, input_shape=(None, dim_f, 1), return_state=True,
                                          recurrent_dropout=0.2, dropout=0.2,
                                          return_sequences=True, name='freq_revers_time')
        out_freq_lstm_flipped = lstm_freq_moudule_backward(
            tf.transpose(freq_vec_input_flipped, (0, 2, 1)))  # run over time samples
        tot_hidden_freq_flipped = tf.expand_dims(out_freq_lstm_flipped[0], axis=2)
        scope.reuse_variables()  # the variables will be reused.
        for i in range(1, dim_t):
            freq_vec_input_flipped = tf.expand_dims(Freq_timeStack_input_fliped[:, i, 0:dim_f], axis=1)

            scope.reuse_variables()  # the variables will be reused.
            out_lstm_freq = lstm_freq_moudule_backward(tf.transpose(freq_vec_input_flipped, (0, 2, 1)))
            tot_hidden_freq_flipped = tf.concat([tot_hidden_freq_flipped, tf.expand_dims(out_lstm_freq[0], axis=2)],
                                                axis=2)
        tot_hidden_freq_flipped = tf.math.cumsum(tot_hidden_freq_flipped, axis=1)

    # RNN 3 -> frequency axis
    # summing over all frequency shared weights outputs of the lstm :
    with tf.variable_scope('freq_lstm') as scope:
        freq_vec_input = tf.expand_dims(Freq_timeStack_input[:, 0, 0:dim_f], axis=1)

        lstm_freq_moudule_forward = LSTM(hidden_layer, input_shape=(None, dim_f, 1), return_state=True,
                                         recurrent_dropout=0.2, dropout=0.2,
                                         return_sequences=True, name='freq_time')
        out_freq_lstm = lstm_freq_moudule_forward(tf.transpose(freq_vec_input, (0, 2, 1)))  # run over time samples
        tot_hidden_freq = tf.expand_dims(out_freq_lstm[0], axis=2)
        scope.reuse_variables()  # the variables will be reused.
        for i in range(1, dim_t):
            freq_vec_input = tf.expand_dims(Freq_timeStack_input[:, i, 0:dim_f], axis=1)

            scope.reuse_variables()  # the variables will be reused.
            out_lstm_freq = lstm_freq_moudule_forward(tf.transpose(freq_vec_input, (0, 2, 1)))
            tot_hidden_freq = tf.concat([tot_hidden_freq, tf.expand_dims(out_lstm_freq[0], axis=2)], axis=2)
        tot_hidden_freq = tf.math.cumsum(tot_hidden_freq, axis=1)
    # Concatination of the 3 LSTMs outputs:
    tot_hidden = tf.transpose(tot_hidden, [0, 2, 1, 3])
    CONCAT_3_RNN_OUPUT = tf.transpose(tf.concat([tot_hidden, tot_hidden_freq_flipped, tot_hidden_freq], axis=2),
                                      (0, 2, 1, 3))
    return CONCAT_3_RNN_OUPUT

def gaussian_mixture_loss(label_spect, alpha, mu, sigma, ):
    # Dimentions: (batch_size,dim_f1,dim_t1,num of mixture)

    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,
            scale=sigma))
    log_loss = -tf.reduce_sum(gm.log_prob(tf.squeeze(label_spect)))
    return log_loss

def MelNET(dim_f, dim_t, hidden_layer=3, batch_norm=0):
    # MElNET Architecture :
    a_session = tf.Session()
    # Trainning parameters:
    learning_rate = tf.placeholder(tf.float32, shape=[])
    mixture_num = 10  # Number of mixuture gaussians
    # Input output for net :
    spectrogram = tf.placeholder(shape=(None, dim_f, dim_t), dtype=tf.float32)
    labels_spect = tf.placeholder(shape=(None, dim_f, dim_t), dtype=tf.float32)
    ## an option is to padd with zeros instead rotating:
    # delay input Freq -> for input to FrequencyDelayedStack
    freq_input_matrix = tf.roll(spectrogram, shift=1, axis=1)
    # delay input Time -> for input to TimeDelayedStack
    Time_input_matrix = tf.roll(spectrogram, shift=1, axis=2)
    # delay input Time -> for input to TimeDelayedStack -> for frequency rnn
    Freq_timeStack_input = tf.transpose(Time_input_matrix, [0, 2, 1])

    # Running the TimeDelayedStack first in order to insert it to FrequencyDelayedStack
    concat_output = TimeDelayedStack(Time_input_matrix, Freq_timeStack_input, dim_f,
                                     dim_t, hidden_layer)
    if batch_norm == 1:
        concat_output = keras.layers.BatchNormalization()(concat_output)

    # Define the matrics matmul weights variable:
    matwh1 = tf.Variable(shape=[dim_t, 3 * dim_t], dtype=tf.float32,
                         initial_value=tf.random_uniform(shape=[dim_t, 3 * dim_t], minval=0, maxval=1))
    ##### Calculating over all hidden layers:
    vec = concat_output[:, :, :, 0]
    Tot_matrix = tf.expand_dims(tf.matmul(matwh1, vec), axis=1)
    for i in range(1, hidden_layer):
        vec = concat_output[:, :, :, i]
        output_mat = tf.expand_dims(tf.matmul(matwh1, vec), axis=1)
        Tot_matrix = tf.concat([Tot_matrix, output_mat], axis=1)

    t_concat = tf.expand_dims(Time_input_matrix, axis=1)
    # arranging the residual part in time to match the size of hidden layers:
    for i in range(1, hidden_layer):
        t_concat = tf.concat([t_concat, tf.expand_dims(Time_input_matrix, axis=1)], axis=1)
    # creating input for frequencyDelayStack for time residual:
    time_output = tf.transpose(Tot_matrix, (0, 1, 3, 2)) + t_concat
    # calculating  frequencyDelayStack :
    out_freq = FrequencyDelayedStack(freq_input_matrix, time_output, dim_f, dim_t, hidden_layer)
    if batch_norm == 1:
        out_freq = keras.layers.BatchNormalization()(out_freq, training=True)
        mu = Conv2D(filters=mixture_num, kernel_size=(1, 1), activation='relu', padding='same')(
            tf.expand_dims(out_freq, axis=3))
    if batch_norm == 1:
        mu = keras.layers.BatchNormalization()(mu)
    sigma = Conv2D(filters=mixture_num, kernel_size=(1, 1), activation='relu', padding='same')(
        tf.expand_dims(out_freq, axis=3))
    if batch_norm == 1:
        sigma = keras.layers.BatchNormalization()(sigma)
    sigma = K.exp(sigma)
    alpha = Conv2D(filters=mixture_num, kernel_size=(1, 1), activation='relu', padding='same')(
        tf.expand_dims(out_freq, axis=3))
    if batch_norm == 1:
        alpha = keras.layers.BatchNormalization()(alpha)
    alpha = K.exp(alpha)
    # mixture sum to one :
    aa = tf.expand_dims(alpha[:, :, :, 0] / K.sum((alpha), axis=3), axis=3)
    for i in range(1, mixture_num, 1):
        aa = tf.concat([aa, tf.expand_dims(alpha[:, :, :, i] / K.sum((alpha), axis=3), axis=3)], axis=3)
    alpha = aa
    # out_put_final = out_conv3
    loss1 = gaussian_mixture_loss(labels_spect, alpha, mu, sigma)

    # deffining ML loss for gmm:
    train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(
        loss1)

    init = tf.global_variables_initializer()
    a_session.run(init)
    return train_op, spectrogram, labels_spect, loss1, a_session, mu, sigma, alpha, learning_rate


# Main runnig the loop train :
if __name__ == "__main__":
    # Creating the graph of MELNET model:
    dim_f1 = 32
    dim_t1 = 50
    hidden_layer1 = 5
    train_op1, spectrogram1, labels_spect1, loss11, session1, miu1, sigma1, alpha1, learning_rate1 = MelNET(dim_f1,
                                                                                                            dim_t1,
                                                                                                            hidden_layer=hidden_layer1,
                                                                                                            batch_norm=1)
    restore = 1
    restore_path = r'ckpt file here'
    if restore:
        saver1 = tf.train.Saver(restore_path)
        saver1.restore(session1,
                       save_path=restore_path)

    # Trainning loop :
    batch_size1 = 35
    num_epochs1 = 500
    num_steps_over_batch1 = 8  # how many repetition for gradient decent over the current batch
    folder_paths = [r"Yours data here"]  # Folder containning the data (wav files)
    lr = 5e-3
    for epoch_num in range(1, num_epochs1, 1):
        folder_path = folder_paths[0]
        Input1 = create_data_train(folder_path)
        Input1 = Input1 - np.min(Input1)
        # Input1 = Input1 / np.max(Input1)
        dim_f1 = np.size(Input1, 1)
        dim_t1 = np.size(Input1, 2)
        if (epoch_num == 10):
            lr = lr / 2
        # Running the train process:
        train_data(Input1, batch_size1, int(np.size(Input1, axis=0) / batch_size1), num_steps_over_batch1,
                   train_op=train_op1,
                   spectrogram=spectrogram1, labels_spect=labels_spect1, loss1=loss11, session=session1,
                   learning_rate1=learning_rate1, lr=lr)
        gc.collect()

    ######################## Generating outputs by sampling the estimated distribution:################################
    count1 = 1
    input_generator = Input1[range(count1 * batch_size1, count1 * batch_size1 + batch_size1), :, :]
    input_generator = input_generator - np.min(input_generator)
    col = 20
    col1 = 20
    input_data2 = np.random.random((batch_size1, dim_f1, 1 * dim_t1)) * np.mean(input_generator) * 0.4
    input_data2[:, :, 0:col] = input_generator[:, :, 0:col]
    miut1, sigmat1, alphat1 = session1.run([miu1, sigma1, alpha1], feed_dict={spectrogram1: input_data2,
                                                                              labels_spect1: input_data2})
    output_from_Generation, output_from_Generation_batch = generating_from_disribution_new(miut1, sigmat1, alphat1)

    # sanity check before generation:
    plt.figure(32)
    plt.imshow((output_from_Generation[:, 50:]))

    # Generation :
    for ik in range(dim_t1 - col):
        for jk in range(3, dim_f1 - 10, 1):
            input_net = input_data2[:, :, 0:dim_t1]
            miut1, sigmat1, alphat1 = session1.run([miu1, sigma1, alpha1], feed_dict={spectrogram1: input_net,
                                                                                      labels_spect1: input_net})
            plt.close()
            output_from_Generation, output_from_Generation_batch = generating_from_disribution_new(miut1, sigmat1,
                                                                                                   alphat1)
            temp_amplitude = output_from_Generation_batch[1:, :, :]
            input_data2[:, jk, col] = temp_amplitude[:, jk, col]
            plt.figure(2)
            plt.imshow(input_net[0, :])
            plt.pause(0.1)
        col = col + 1
    new = np.zeros((dim_f1, dim_t1))
    for ii in range(batch_size1):
        new = np.append(new, input_data2[ii, :, col1 - 20:col1 + 1], axis=1)
    OUT = new[:, 50:]
    plt.figure(4)
    plt.imshow(OUT)
    rate = 24000
    FS = rate
    AUDIO_SIGNAL = return_to_audio(FS, OUT[:, 1:])
