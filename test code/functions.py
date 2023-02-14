import streamlit as st
import numpy as np
# import pandas as pd
import librosa
from scipy.fft import rfft, rfftfreq ,irfft


@st.cache
# Function to read audio file and return siganl y_axis time x_axis and sampling rate 
def read_sound_file(file):
    signal , sample_rate = librosa.load(file, sr=None)
    #1d file 
    if signal.ndim==1:
        time =np.linspace(0,signal.shape[0]/sample_rate,signal.shape[0] )
    #2d file conversion
    else:
        signal_1=[]
        signal_2=[]
        for i in range (signal.shape[0]):
            signal_1.append(signal[i][0])
            signal_2.append(signal[i][1])
        signal_1=np.array(signal_1)
        signal_2=np.array(signal_2)    
        signal=signal_1+signal_2
        time =np.linspace(0,signal.shape[0]/sample_rate,signal.shape[0])
    return [signal,time,sample_rate]

# Function to take time domain signal and return Fourier domain signal(power,phase,frequency)
def generate_fourier(signal,sample_rate):
    rfft_file = rfft(st.session_state.signal)
    power = np.abs(rfft_file)
    phase = np.angle(rfft_file)
    frequency = rfftfreq(len(signal), 1/sample_rate)
    # The maximum frequency is half the sample rate
    points_per_freq = len(frequency) / (sample_rate / 2)
    return[power,phase,frequency,points_per_freq]