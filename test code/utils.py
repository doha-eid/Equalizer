import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile



def read_wav(file):
    try:
        sample_rate, signal = wavfile.read(file)
        time= np.linspace(0,signal.shape[0]/sample_rate,signal.shape[0] )
        return [signal, time,sample_rate]
    except:
        time=np.linspace(0,5,2000)
        full_signals=np.zeros(time.shape)
        return full_signals, time


