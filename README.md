# MelNet-Tensorflow(WIP)
## Introduction
implementation of melnet(FAIR) using tensorflow https://arxiv.org/pdf/1906.01083.pdf
Generation of singing voice(Unconditional) , Data was taken from youtube and was converted to wav files.
The model was trained on 1 singer Ariana Grande, 2 hours of singing.

### data set:
Fs: 48[KHz] -> donwsampled to 24[KHz].
- Melspectrogram parameters: 

   fft_size = 2048  -> window size for the FFT

   n_mel_freq_components = 32  -> number of mel frequency channels

   start_freq = 300  -> Hz What frequency to start sampling our melS from

   end_freq = 8000  -> Hz  What frequency to stop sampling our melS from

   Time_window = 50  -> Time window to slice the spectrograms

   Each input contains matrix of size: (32,50)


### Train:
RMSPOP optimizer with lr: 7e-4 , momentum: 1e-4.
Every 10 epochs learning rate was down in half-> lr/2
Tested with batch size:16
### Generations (Current results):
![](./images/generations.PNG)
### Time domain signal
![](./images/TimeSignal.png)

#### MelNet.py:
- Graph creation:

   def FrequencyDelayedStack 

   def TimeDelayedStack

   def MelNET -> creation of the graph.

   def gaussian_mixture_loss 

- generating sample by sample using GMM :

   def generating_from_disribution_new


** At the moment the generation process takes a lot of time, in the next few weeks intend to improve efficiency, and upload audio 
results.


