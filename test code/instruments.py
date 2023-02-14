import streamlit as st
import  streamlit_vertical_slider  as svs

import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy.io.wavfile import write

from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
import plotly.graph_objects as go
import streamlit.components.v1 as components

import librosa

import wave

st.set_page_config(
    page_title="Music",
    layout="wide")


if "slider1" not in st.session_state:
    st.session_state.slider1=0
if "slider2" not in st.session_state:
    st.session_state.slider2=0
if "slider3" not in st.session_state:
    st.session_state.slider3=0
if "slider4" not in st.session_state:
    st.session_state.slider4=0
# if "slider5" not in st.session_state:
#     st.session_state.slider5=0
# if "slider6" not in st.session_state:
#     st.session_state.slider6=0
# if "slider7" not in st.session_state:
#     st.session_state.slider7=0
# if "slider8" not in st.session_state:
#     st.session_state.slider8=0
# if "slider9" not in st.session_state:
#     st.session_state.slider9=0
# if "slider10" not in st.session_state:
#     st.session_state.slider10=0 

if "target_idx_1" not in st.session_state :
    st.session_state.target_idx_1=0
if "target_idx_2" not in st.session_state :
    st.session_state.target_idx_2=0
if "target_idx_3" not in st.session_state :
    st.session_state.target_idx_3=0

# Initialization of Session State attributes (time,uploaded_signal)
if 'time' not in st.session_state:
    st.session_state.time =np.linspace(0,10,2000)
if 'signal' not in st.session_state:
    st.session_state.signal = np.sin(2*np.pi*st.session_state.time)
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = 1

if "yf_array" not in st.session_state :
    st.session_state.yf_array=rfft(st.session_state.signal)
if "xf_array" not in st.session_state :
    st.session_state.xf_array=rfftfreq(st.session_state.sample_rate * int(st.session_state.time[-1]), 1 / st.session_state.sample_rate)

time=np.linspace(0,5,2000)
signal=np.zeros(time.shape)



#Upload wav file
file=st.file_uploader(label="Upload Signal File", key="uploaded_file",type=["csv","wav"])
if file :
    if file.name.split(".")[-1]=="wav":
        signal , sample_rate = librosa.load(file, sr=None)
        # sample_rate, signal = wavfile.read(file) 
        if signal.ndim==1:
            length = signal.shape[0] / sample_rate
            # time = np.linspace(0., length, signal.shape[0])
            time =np.linspace(0,signal.shape[0]/sample_rate,signal.shape[0] )
            st.session_state.signal=signal
            st.session_state.time= time
            st.session_state.sample_rate= sample_rate
        else:
            #2d file conversion
            signal_1=[]
            signal_2=[]
            for i in range (signal.shape[0]):
                signal_1.append(signal[i][0])
                signal_2.append(signal[i][1])
            signal_1=np.array(signal_1)
            signal_2=np.array(signal_2)    
            signal=signal_1+signal_2

            length = signal.shape[0] / sample_rate
            # time = np.linspace(0., length, signal.shape[0])
            time =np.linspace(0,signal.shape[0]/sample_rate,signal.shape[0] )
            st.session_state.signal=signal
            st.session_state.time= time
            st.session_state.sample_rate= sample_rate

        #display audio file
        # audio_bytes=file.read()
        st.audio(file,format='audio/wav')



#Calculate FOurier Transform
N = st.session_state.sample_rate * int(st.session_state.time[-1])
# yf = rfft(st.session_state.signal)
# xf = rfftfreq(N, 1 / st.session_state.sample_rate)
# phase=np.angle(st.session_state.yf_array)
# st.session_state.yf_array=yf
# st.session_state.xf_array=xf

# Fourier phase and amplitude
rfft_file = rfft(st.session_state.signal)
amplitude = np.abs(rfft_file)
phase = np.angle(rfft_file)
# frequency = rfftfreq(len(st.session_state.signal), 1/sample_rate)
frequency = rfftfreq(N, 1/sample_rate)

st.session_state.xf_array=frequency

if "amplitude" not in st.session_state :
    st.session_state.amplitude=amplitude



# The maximum frequency is half the sample rate
points_per_freq = len(frequency) / (st.session_state.sample_rate / 2)

#upload and view signal
time_signal_graph, fourier_signal_graph  = st.columns([ 3, 3 ])

# sliders columns
# slider_1, slider_2, slider_3, slider_4, slider_5, slider_6, slider_7, slider_8, slider_9, slider_10 ,check_boxes = st.columns([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
slider_1, slider_2, slider_3,slider_4 ,check_boxes = st.columns([4, 4, 4,4,4])

with slider_1 :
    
    s1= svs.vertical_slider(key="slider1", default_value=0, step=0.1, min_value=0, 
                    max_value=5,   slider_color= 'green',#optional
                    track_color='lightgray', #optional
                    thumb_color = 'red' #optional,
                    )

    # st.session_state.amplitude[(frequency>1) & (frequency < 500) ] = amplitude[(frequency>1) & (frequency < 500) ] * (st.session_state.slider2)

    st.session_state.amplitude[int(0*points_per_freq):int(500*points_per_freq)]=  amplitude[int(0*points_per_freq):int(500*points_per_freq)]*(st.session_state.slider1)    

    # st.session_state.yf_array[int(0*points_per_freq):int(500*points_per_freq)]=  st.session_state.yf_array[int(0*points_per_freq):int(500*points_per_freq)]*(st.session_state.slider1)    

    
    # for i in range (10,499):
    #     st.session_state.yf_array[int(i*points_per_freq)]=0
    # st.write(st.session_state.yf_array[int(i*points_per_freq)])
    st.write("Drums")


with slider_2 :
    s2= svs.vertical_slider(key="slider2", default_value=0, step=0.1, min_value=0, 
                    max_value=5,   slider_color= 'red',#optional
                    track_color='lightgray', #optional
                    thumb_color = 'green' #optional
                    )

    # st.session_state.amplitude[(frequency>500) & (frequency < 1000) ] = amplitude[(frequency>500) & (frequency < 1000) ] * (st.session_state.slider2)

    st.session_state.amplitude[int(500*points_per_freq):int(1000*points_per_freq)]=  amplitude[int(500*points_per_freq):int(1000*points_per_freq)]*(st.session_state.slider2)    

    # st.session_state.yf_array[int(500*points_per_freq):int(1000*points_per_freq)]=  st.session_state.yf_array[int(500*points_per_freq):int(1000*points_per_freq)]*(st.session_state.slider2)    

    # st.write(st.session_state.yf_array[int(i*points_per_freq)]
    st.write("drums")

with slider_3 :
    s3= svs.vertical_slider( key="slider3", default_value=0  , step=0.1, min_value=0, 
    
                    max_value=5,   slider_color= 'red',#optional
                    track_color='green', #optional
                    thumb_color = 'blue' #optional
                    )

    # st.session_state.amplitude[(frequency>1000) & (frequency < 5000) ] = amplitude[(frequency>1000) & (frequency < 5000) ] * (st.session_state.slider3)

    st.session_state.amplitude[int(1000*points_per_freq):int(5000*points_per_freq)]=  amplitude[int(1000*points_per_freq):int(5000*points_per_freq)]*(st.session_state.slider3)    

    # st.session_state.yf_array[int(1000*points_per_freq):int(5000*points_per_freq)]=  st.session_state.yf_array[int(1000*points_per_freq):int(5000*points_per_freq)]*(st.session_state.slider3)    

    st.write("xylophone")

with slider_4 :
    s3= svs.vertical_slider( key="slider4", default_value=0  , step=1, min_value=0, 
    
                    max_value=15,   slider_color= 'red',#optional
                    track_color='green', #optional
                    thumb_color = 'blue' #optional
                    )

    # st.session_state.amplitude[(frequency>1000) & (frequency < 5000) ] = amplitude[(frequency>1000) & (frequency < 5000) ] * (st.session_state.slider3)

    # st.session_state.amplitude[int(1000*points_per_freq):int(5000*points_per_freq)]=  amplitude[int(1000*points_per_freq):int(5000*points_per_freq)]*(st.session_state.slider3)    

    # st.session_state.yf_array[int(1000*points_per_freq):int(5000*points_per_freq)]=  st.session_state.yf_array[int(1000*points_per_freq):int(5000*points_per_freq)]*(st.session_state.slider3)    

    st.write("Volume")


with check_boxes:
    frequency = st.checkbox('frequency', value= True)  
    instruments = st.checkbox('Musical instruments',value=False) 
    vowels = st.checkbox('Vowels', value= False)  
    medical = st.checkbox('Medical',value=False) 
    elective = st.checkbox('Elective',value=False) 


# #Dynamic Plot
# y=[]
# x=[]
# for i in range(0,500):
#         x.append(time[i])
#         y.append(signal[i])
# count=500
# def draw_graph(i):
#     global count
#     count +=500
#     for i in range(count-500,count):
#         x.append(time[i])
#         y.append(signal[i])
#         x.pop(0)
#         y.pop(0)
#     plt.cla()
#     plt.plot(x,y)

fourier_signal = np.multiply(st.session_state.amplitude, np.exp(1j*phase))
new_sig = irfft(fourier_signal)

# new_sig = irfft(st.session_state.yf_array)
# norm_new_sig = np.int16(new_sig * (32767 / new_sig.max()))
Hear_signal= new_sig *  st.session_state.slider4
write("new_piano.wav", sample_rate, Hear_signal)
st.audio("new_piano.wav",format='audio/wav')

#column to draw time graph
with time_signal_graph:
    
    if file:
        full_signals, time= st.session_state.signal, st.session_state.time
        # line_ani = animation.FuncAnimation(plt.gcf(),draw_graph,interval=1 )
        # components.html(line_ani.to_jshtml(), height=1000)
    else:
        time= np.linspace(0, 4, 2000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.time[0:1000],
                                y=st.session_state.slider4*st.session_state.signal[0:1000],
                                mode='lines',
                                name='Signal'))
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig,use_container_width=True)

#column to draw fourier graph
with fourier_signal_graph:
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.xf_array,
                                y=st.session_state.amplitude,
                                mode='lines',
                                name='fourier'))
    fig.update_xaxes(showgrid=True, zerolinecolor='black', gridcolor='lightblue', range = (-0.1,25000))
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig,use_container_width=True)