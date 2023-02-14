import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import time as tim
import librosa
import librosa.display
from scipy.fft import rfft, rfftfreq ,irfft

def read_audio_file(file):
    original_signal , sample_rate = librosa.load(file, sr=None)
    time = np.linspace(0,original_signal.shape[0]/sample_rate,original_signal.shape[0] )
    return [original_signal,sample_rate,time]

def fourier_tranform(original_signal,sample_rate,time):
    N = sample_rate * int(time[-1])
    # Fourier phase and amplitude
    rfft_file = rfft(original_signal)
    amplitude = np.abs(rfft_file)
    phase = np.angle(rfft_file)
    # frequency = rfftfreq(N, 1/sample_rate)
    frequency = rfftfreq(len(original_signal), 1/sample_rate)
    points_per_freq = len(frequency) / (sample_rate / 2)
    return [amplitude,phase,frequency,points_per_freq]

# First graph
def plotShow(data, idata,resume_btn,sr):
    time1 = len(data)/(sr)
    if time1>1:
        time1 = int(time1)
    time1 = np.linspace(0,time1,len(data))   
    df = pd.DataFrame({'time': time1[::300], 
                        'amplitude': data[:: 300],
                        'amplitude after processing': idata[::300]}, columns=[
                        'time', 'amplitude','amplitude after processing'])
    N = df.shape[0]  # number of elements in the dataframe
    burst = 10      # number of elements (months) to add to the plot
    size = burst 
    step_df = df.iloc[:st.session_state.size1]
    if st.session_state.size1 ==0:
        step_df = df.iloc[0:N]
    lines = plot_animation(step_df)
    line_plot = st.altair_chart(lines)
    line_plot= line_plot.altair_chart(lines)
    
    if resume_btn: 
        st.session_state.flag = not(st.session_state.flag)
        if st.session_state.flag :
            for i in range( st.session_state.start,N):
                st.session_state.start =i 
                step_df = df.iloc[size:size+i]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                st.session_state.size1 = size
                size = i + burst
                tim.sleep(.1)
    
    if st.session_state.flag :
        for i in range(st.session_state.start,N):
                st.session_state.start =i 
                step_df = df.iloc[size:size+i]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                st.session_state.size1 = size
                size = i + burst
                tim.sleep(.1)


# Second Graph
def plot_animation(df):
    brush = alt.selection_interval()
    chart1 = alt.Chart(df).mark_line().encode(
            x=alt.X('time', axis=alt.Axis(title='Time')),
        ).properties(
            width=400,
            height=150
        ).add_selection(
            brush).interactive()
    figure = chart1.encode(
                y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude')))| chart1.encode(
                y=alt.Y('amplitude after processing',axis=alt.Axis(title='Amplitude after'))).add_selection(
            brush)
    return figure

def voice_changer(signal,sample_rate,Num_of_steps):
    voice_changed=librosa.effects.pitch_shift(signal,sr=sample_rate,n_steps=Num_of_steps)
    return voice_changed