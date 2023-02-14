import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import  streamlit_vertical_slider  as svs

import time as tt

st.set_page_config(
    page_title="Plot",
    # layout="wide"
    )

if "slider1" not in st.session_state:
    st.session_state.slider1=0
if "slider2" not in st.session_state:
    st.session_state.slider2=0

if "time" not in st.session_state:
    st.session_state.time=0
if "signal" not in st.session_state:
    st.session_state.signal=0

# if "amplitude" not in st.session_state:
#     st.session_state.amplitude=1

# main_container = st.container()
# placeholder = st.empty()
def update_signal () :
    st.session_state.signal=st.session_state.slider1*np.sin(2*np.pi*freq*time)


time=np.linspace(0,5,2000)
freq=10
st.session_state.signal=st.session_state.slider1*np.sin(2*np.pi*freq*time)

graph_1, graph_2 =st.columns([4,4])

with graph_1:
    # Create Graph
    fig = plt.figure(figsize=(6,2))
    plt.plot(time,st.session_state.signal)
    st.pyplot(fig)

with graph_2:
    # Create Graph
    fig = plt.figure(figsize=(6,2))
    plt.plot(time,st.session_state.signal)
    st.pyplot(fig)


slider_1, slider_2 =st.columns([4,4])
# Add Sliders
with slider_1:
    s1= svs.vertical_slider(key="slider1", default_value=0, step=1,
                        min_value=0,   max_value=100)
    # if s1:
    update_signal()

with slider_2:
    # s2= svs.vertical_slider(key="slider2", default_value=0, step=1,
    #     min_value=0,   max_value=100)
    s2=st.slider("Slider2",min_value=0,step=1,max_value=100,key="slider2",on_change=update_signal)  
    # st.session_state.amplitude=s2
    # main_container.write("Equalizer")
    # time=np.linspace(0,5,2000)
    # freq=10
    # signal=st.session_state.amplitude*np.sin(2*np.pi*freq*time)
    # fig = plt.figure(figsize=(6,2))
    # plt.plot(time,signal)
    # placeholder.pyplot(fig)

    # main_container.pyplot(fig)


#                 # on_change =main_countainer.st.write(5)
#                 # , on_change=update_graph()
#                 # ,continuous_update=False
#     st.session_state.amplitude=s1

    # time=np.linspace(0,5,2000)
    # freq=10
    # signal=st.session_state.amplitude*np.sin(2*np.pi*freq*time)
    # fig = plt.figure(figsize=(6,2))
    # plt.plot(time,signal)
    # placeholder.pyplot(fig)

    # main_container.pyplot(fig)
    # main_container.write("Equalizer")
# @st.cache(show_spinner=False)
# def run_sum(a, b):
#     # tt.sleep(1)
#     return a + b
# Create signal
# time=np.linspace(0,5,2000)
# freq=10
# signal=st.session_state.amplitude*np.sin(2*np.pi*freq*time)
# fig = plt.figure(figsize=(6,2))
# plt.plot(time,signal)
# main_container.pyplot(fig)
# main_container.write(st.session_state.amplitude)

# main_container.write(s1)