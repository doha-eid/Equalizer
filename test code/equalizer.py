import streamlit as st
import  streamlit_vertical_slider  as svs

import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
import pandas as pd

from scipy.io import wavfile

from scipy.fft import rfft, rfftfreq ,irfft
import plotly.graph_objects as go
import streamlit.components.v1 as components

import librosa
from functions import read_sound_file ,generate_fourier

# ------------------------------------------ Initialization of Session States------------------------------------------------------------#

# Time domain attributes (time(x_axis),signal(y_axis))
if 'time' not in st.session_state:
    st.session_state.time =np.linspace(0,5,2000)
if 'signal' not in st.session_state:
    st.session_state.signal = np.zeros(st.session_state.time.shape)
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = 1

# Fourier domain attributes (frequency(x_axis),power)
if "frequency" not in st.session_state :
    st.session_state.frequency=np.linspace(0,5,2000)
if "power" not in st.session_state :
    st.session_state.power=np.zeros(st.session_state.frequency.shape)

# Initialization of Fourier domain signal values
power=np.zeros(st.session_state.frequency.shape)
points_per_freq = len(st.session_state.frequency) / (st.session_state.sample_rate / 2)
phase = np.zeros(st.session_state.frequency.shape)
# Initialization odf orignal time domain signal values
time=np.linspace(0,5,2000)
signal=np.zeros(time.shape)

st.set_page_config(
    page_title="Final Equalizer",
    layout="wide")

# with open("style.css") as design:
#     st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)

# ---------------------------------SIDE BAR------------------------------------------------------#
with st.sidebar:
#Upload audio file
    file=st.file_uploader(label="Upload Signal File", key="uploaded_file",type=["csv","wav"])
    if file :
        if file.name.split(".")[-1]=="wav":
            # GET time domain signal
            signal,time,sample_rate=read_sound_file(file)
            st.session_state.signal=signal
            st.session_state.time= time
            st.session_state.sample_rate= sample_rate
            # ----------------------------------------------------Calculate Fourier Transform----------------------------------------#
            # GET Fourier domain signal
            power, phase, frequency, points_per_freq = generate_fourier(signal,sample_rate)
            st.session_state.frequency=frequency
            st.session_state.power=power
            # The maximum frequency is half the sample rate
            points_per_freq = len(frequency) / (st.session_state.sample_rate / 2)
    # Radio Button to chose signal mode
    # signal_options = st.radio("Choose siganl mode", ( "Frequency", "Music", "Vowels", "ECG" ) )
    # Select Box to chose signal mode
    option = st.selectbox("Choose siganl mode", ( "Frequency", "Vowels", "Music",  "ECG" ) )
    st.write('You selected:', option)


#Display audio file
# st.audio(file,format='audio/wav')
# #--------------------------Graph Columns---------------------------------------------------------#
time_signal_graph, fourier_signal_graph  = st.columns(2)


# -----------------------------------Volume Slider--------------------------#
if "time_amplitude" not in st.session_state :
    st.session_state.time_amplitude=1
st.slider("Volume", min_value=0, max_value=10, value=1, step=1, key="time_amplitude")

#--------------------------SLIDERS Columns---------------------------------------------------------#
sliders_numbers=10
sliders_columns=st.columns(sliders_numbers)

band_width=[0,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000]

# slider_frequency_range=[[0,2000]]

for idx,sliders_column in enumerate(sliders_columns) :
    # slider_key=f"slider{idx+1}"
    # generate session state of sliders
    if  f"slider{idx+1}" not in st.session_state :
        st.session_state[f"slider{idx+1}"]=1
    # generate vertical sliders
    with sliders_column:
        s1= svs.vertical_slider(key=f"slider{idx+1}", default_value=1, 
                                min_value=0, step=0.1, max_value=100)
        st.write(f"{int(band_width[idx]/1000)}-{int(band_width[idx+1]/1000)} KHz")
# ----------------------------------------------------UPDATE Fourier Data From SLIDERS----------------------------------------#
        st.session_state.power[int(band_width[idx]*points_per_freq):int(band_width[idx+1]*points_per_freq)]=  power[int(band_width[idx]*points_per_freq):int(band_width[idx+1]*points_per_freq)]*(st.session_state[f"slider{idx+1}" ])   
        # st.write(idx)
        # st.write(f"slider{idx+1}") 
        # st.write(st.session_state[f"slider{idx+1}"]) 
        # st.write(st.session_state.power[int(band_width[idx]*points_per_freq):int(band_width[idx+1]*points_per_freq)]) 


# ----------------------------------------------------Calculate INVERSE Fourier Transform----------------------------------------#
fourier_signal = np.multiply(st.session_state.power, np.exp(1j*phase))
inverse_fourier_sig = irfft(fourier_signal)
st.session_state.signal=inverse_fourier_sig

# ---------------------------------------DRAW GRAPHS-----------------------------------------------#

#--------------------------Graph Columns---------------------------------------------------------#
# time_signal_graph, fourier_signal_graph  = st.columns(2)


#Time Domain Graph Column
with time_signal_graph:
    # line_ani = animation.FuncAnimation(plt.gcf(),draw_graph,interval=500 )
    # components.html(line_ani.to_jshtml(), height=1000)
    fig = go.Figure()
    # Draw Original Time Signal
    fig.add_trace(go.Scatter(x=time, y=signal,
                                mode='lines', name='Original Signal',line={"color":"orange"}))
    # Draw Updated Time Signal
    fig.add_trace(go.Scatter(x=st.session_state.time, y=st.session_state.time_amplitude*st.session_state.signal, mode='lines',
                            name='New Signal',marker=dict(size=10)))
    fig.update_xaxes(showgrid=True, zerolinecolor='blue', gridcolor='lightblue', range = (-0.1,st.session_state.time[-1]))
    fig.update_yaxes(showgrid=True, zerolinecolor='blue', gridcolor='lightblue', 
                    range = ((-1*(max(st.session_state.signal)+0.1*max(st.session_state.signal))*10),((max(st.session_state.signal)+0.1*max(st.session_state.signal))*10)))
    # fig.update_layout(
    #         font = dict(size = 20),
    #         xaxis_title="Time (sec)",
    #         yaxis_title="Amplitude",
    #         height = 600,
    #         # margin=dict(l=0,r=0,b=5,t=0),
    #         legend=dict(orientation="h",
    #                     yanchor="bottom",
    #                     y=0.92,
    #                     xanchor="right",
    #                     x=0.99,
    #                     font=dict(size= 18, color = 'black'),
    #                     bgcolor="LightSteelBlue"
    #                     ),
    #         paper_bgcolor='rgb(4, 3, 26)',
    #         plot_bgcolor='rgba(255,255,255)'
    #     )
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig,use_container_width=True)

#Fourier Spectogram Graph Column
with fourier_signal_graph:
    # Spectogram
    fig,ax=plt.subplots()
    ax.specgram(st.session_state.signal[0:],NFFT=5000,Fs=st.session_state.sample_rate,cmap='jet')
    # ax.specgram(inverse_fourier_sig,NFFT=5000,Fs=st.session_state.sample_rate,cmap='jet')
    st.pyplot(fig)


    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=st.session_state.frequency,
    #                             y=st.session_state.power,
    #                             mode='lines',
    #                             name='Fourier'))
    # fig.update_xaxes(showgrid=True, zerolinecolor='black', gridcolor='lightblue', range = (-0.1,25000))
    # fig.update_yaxes(automargin=True)
    # st.plotly_chart(fig,use_container_width=True)

wavfile.write("new_audio.wav",st.session_state.sample_rate,inverse_fourier_sig.astype(np.int16))
#Display audio file
st.audio("new_audio.wav",format='audio/wav')