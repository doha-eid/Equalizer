import streamlit as st
import  streamlit_vertical_slider  as svs

import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
import plotly.graph_objects as go
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Equalizer",
    layout="wide")


if "slider1" not in st.session_state:
    st.session_state.slider1=0
if "slider2" not in st.session_state:
    st.session_state.slider2=0
if "slider3" not in st.session_state:
    st.session_state.slider3=0
# if "slider4" not in st.session_state:
#     st.session_state.slider4=0
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
if file:
    if file.name.split(".")[-1]=="wav":
        # signal, time=read_wav(file)
        sample_rate, signal = wavfile.read(file) 
        length = signal.shape[0] / sample_rate
        # time = np.linspace(0., length, signal.shape[0])
        time =np.linspace(0,signal.shape[0]/sample_rate,signal.shape[0] )
        st.session_state.signal=signal
        st.session_state.time= time
        st.session_state.sample_rate= sample_rate

#Calculate FOurier Transform
N = st.session_state.sample_rate * int(st.session_state.time[-1])
yf = rfft(st.session_state.signal)
xf = rfftfreq(N, 1 / st.session_state.sample_rate)
st.session_state.yf_array=yf
st.session_state.xf_array=xf

#Filtering the Signal
# The maximum frequency is half the sample rate
points_per_freq = len(xf) / (st.session_state.sample_rate / 2)
# Our target frequency is 400 Hz
st.session_state.target_idx_1 = int(points_per_freq * 400)
# Our target frequency is 600 Hz
st.session_state.target_idx_2 = int(points_per_freq * 600)
# Our target frequency is 2000  Hz
st.session_state.target_idx_3 = int(points_per_freq * 2000)


#upload and view signal
time_signal_graph, fourier_signal_graph  = st.columns([ 3, 3 ])

# sliders columns
# slider_1, slider_2, slider_3, slider_4, slider_5, slider_6, slider_7, slider_8, slider_9, slider_10 ,check_boxes = st.columns([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

slider_1, slider_2, slider_3 ,check_boxes = st.columns([4, 4, 4,4])

with slider_1 :
    
    s1= svs.vertical_slider(key="slider1", default_value=0, step=1, min_value=0, 
                    max_value=100,   slider_color= 'green',#optional
                    track_color='lightgray', #optional
                    thumb_color = 'red' #optional,
                    )
    st.session_state.yf_array[st.session_state.target_idx_1]=st.session_state.slider1*-1j
    st.write(st.session_state.yf_array[st.session_state.target_idx_1])
    st.write("Slider]")

with slider_2 :
    s2= svs.vertical_slider(key="slider2", default_value=0, step=1, min_value=0, 
                    max_value=220500,   slider_color= 'red',#optional
                    track_color='lightgray', #optional
                    thumb_color = 'green' #optional
                    )
    st.session_state.yf_array[st.session_state.target_idx_2]=st.session_state.slider2*1j
    st.write(st.session_state.yf_array[st.session_state.target_idx_2])

with slider_3 :
    s3= svs.vertical_slider( key="slider3", default_value=0  , step=1, min_value=0, 
                    max_value=220500,   slider_color= 'red',#optional
                    track_color='green', #optional
                    thumb_color = 'blue' #optional
                    )
    st.session_state.yf_array[st.session_state.target_idx_3]=st.session_state.slider3*-1j
    st.write(st.session_state.yf_array[st.session_state.target_idx_3])

# with slider_4 :
#     s4= svs.vertical_slider( key="slider4", default_value=10, step=1, min_value=0, 
#                     max_value=100,   slider_color= 'red',#optional
#                     track_color='green', #optional
#                     thumb_color = 'blue' #optional
#                     )

# with slider_5 :
#     s5= svs.vertical_slider( key="slider5", default_value=10, step=1, min_value=0, 
#                     max_value=100,   slider_color= 'red',#optional
#                     track_color='green', #optional
#                     thumb_color = 'blue' #optional
#                     )

# with slider_6 :
#     s1= svs.vertical_slider(key="slider6", default_value=50, step=1, min_value=0, 
#                     max_value=100,   slider_color= 'green',#optional
#                     track_color='lightgray', #optional
#                     thumb_color = 'red' #optional
#                     )

# with slider_7 :
#     s2= svs.vertical_slider(key="slider7", default_value=10, step=1, min_value=0, 
#                     max_value=100,   slider_color= 'red',#optional
#                     track_color='lightgray', #optional
#                     thumb_color = 'green' #optional
#                     )

# with slider_8 :
#     s3= svs.vertical_slider( key="slider8", default_value=10, step=1, min_value=0, 
#                     max_value=100,   slider_color= 'red',#optional
#                     track_color='green', #optional
#                     thumb_color = 'blue' #optional
#                     )

# with slider_9 :
#     s4= svs.vertical_slider( key="slider9", default_value=10, step=1, min_value=0, 
#                     max_value=100,   slider_color= 'red',#optional
#                     track_color='green', #optional
#                     thumb_color = 'blue' #optional
#                     )

# with slider_10 :
#     s5= svs.vertical_slider( key="slider10", default_value=10, step=1, min_value=0, 
#                     max_value=100,   slider_color= 'red',#optional
#                     track_color='green', #optional
#                     thumb_color = 'blue' #optional
#                     )

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
                                y=np.abs(st.session_state.signal[0:1000]),
                                mode='lines',
                                name='Signal'))
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig,use_container_width=True)

#column to draw fourier graph
with fourier_signal_graph:
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.xf_array,
                                y=np.abs(st.session_state.yf_array),
                                mode='lines',
                                name='fourier'))
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig,use_container_width=True)
