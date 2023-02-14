import streamlit as st
import  streamlit_vertical_slider  as svs
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write
from scipy.fft import rfft, rfftfreq ,irfft
from Functions import *

st.set_page_config(
    page_title="Equalizer",
    layout="wide")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -------------------------------------Open css file---------------------------------------------------------#
with open('src\style.css') as sc:
    st.markdown(f"<style>{sc.read()}</style>",unsafe_allow_html=True)

#----------------------------------------Dynamic Session States--------------------------------#
if 'start' not in st.session_state:
    st.session_state['start']=0
if 'size1' not in st.session_state:
    st.session_state['size1']=0
if 'flag' not in st.session_state:
    st.session_state['flag'] = 0

time =np.linspace(0,5,2000)
original_signal = np.zeros(time.shape)
updated_signal = np.zeros(time.shape)
sample_rate= 44100
points_per_freq = 1


upload_file_flag =False
# ---------------------------------SIDE BAR------------------------------------------------------#
with st.sidebar:
    #----------------------------------Upload audio file---------------------------------------#
    file=st.file_uploader(label="Upload Signal File", key="uploaded_file",type=["csv","wav"])
    if file :
        upload_file_flag =True
        file_uploaded =file
        #----------------------------------------Read audio file 
        original_signal , sample_rate , time = read_audio_file(file)
        #----------------------------------------------------Calculate Fourier Transform----------------------------------------#
        amplitude,phase,frequency,points_per_freq = fourier_tranform(original_signal,sample_rate,time)
        st.session_state.amplitude=amplitude
    # ------------------------------------------------ Select Box to chose signal MODE---------------------------------------#
    option = st.selectbox("Choose Signal Mode", ( "Frequency", "Vowels", "Music", "Animal Changer" , "Voice Changer" ) )
    if option == "Frequency":
        band_width={"0HZ-2KHz":[100,2000],"2-4 KHz":[2000,4000],"4-6 KHz":[4000,6000],"6-8 KHz":[6000,8000],"8-10 KHz":[8000,10000],
        "10-12 KHz":[10000,12000],"12-14 KHz":[12000,14000],"14-16 KHz":[14000,16000],"16-18 KHz":[16000,18000],"18-20 KHz":[18000,20000]}
    elif option == "Vowels":
        band_width={"S Vowel":[4000,10000],"Q Vowel":[650,2700],"Y Vowel":[490,1800],}
    elif option == "Music":
        band_width={"Trumpet":[0,500],"Xylophone":[500,1200],"Brass":[1200,7000]}
    elif option == "Animal Changer":
        band_width={"Dog":[187.5,1300],"Horse":[1300,3300],"Duck":[1300,7000]}
    spectogram_checkbox = st.checkbox('Spectogram')

#------------------------------------------------ Position of Graphs Container UP-----------------------------#
time_graphs_container = st.container()
#-------------------------------------------------Position of Audio Columns-----------------------------------#
original_audio_column , new_audio_slider =st.columns(2)
#--------------------------------------------------Position of Spectograms Columns---------------------------------------#
old_spectrogram, new_spectrogram =st.columns(2)

#--------------------------------------Sliders of 3 Modes (Frequency, Vowels, Music)----------------------------------#
if option in ["Frequency", "Vowels", "Music", "Animal Changer"] :
    #--------------------------SLIDERS Columns---------------------------------------------------------#
    sliders_numbers=len(band_width)
    sliders_columns=st.columns(sliders_numbers)
    for idx,band_width_key in enumerate(band_width):
        if  f"slider{idx+1}" not in st.session_state :
            st.session_state[f"slider{idx+1}"]=1
        #-----------------------------------------------------Generate vertical sliders-----------------------------------#
        with sliders_columns[idx]:
            svs.vertical_slider(key=f"slider{idx+1}", default_value=1, 
                                    min_value=0, step=1, max_value=3)
            st.write(f"{band_width_key}")
            if upload_file_flag :
                # ----------------------------------------------------UPDATE Fourier Data From SLIDERS----------------------------------------#
                st.session_state.amplitude[int(band_width[band_width_key][0]*points_per_freq):int(band_width[band_width_key][1]*points_per_freq)] =  amplitude[int(band_width[band_width_key][0]*points_per_freq):int(band_width[band_width_key][1]*points_per_freq)]*(st.session_state[f"slider{idx+1}"])   
    # check if file is uploaded make inverse fourier
    if upload_file_flag :
        # ----------------------------------------------------Calculate INVERSE Fourier Transform----------------------------------------# 
        fourier_signal = np.multiply(st.session_state.amplitude, np.exp(1j*phase))
        updated_signal = irfft(fourier_signal)
        write("new_audio.wav", sample_rate, updated_signal)
#-------------------------------------------------------------Voice Changer Mode----------------------------------------------#
elif option == "Voice Changer": 
    #----------------------------------Radio buttons to select sounnd tone---------------------------#
    sound_option = st.radio("Voice Change",('Male', 'Female', 'Child'))
    if sound_option == 'Male':
        shift = -3
    elif sound_option == 'Female':
        shift = 0
    elif sound_option == 'Child':
        shift = 5
    if upload_file_flag :
        updated_signal = voice_changer( original_signal, sample_rate,shift)
        write("new_voice.wav", sample_rate, updated_signal)

#-------------------------------------------------Draw Dynamic Time Graph-----------------------------------#
with time_graphs_container:
    resume= st.button('Play/Pause')
    plotShow(original_signal[:len(updated_signal)],updated_signal,resume,sample_rate )

# Check if spectogram checkbox is presssed Draw Spectrogram
if spectogram_checkbox:
    #-------------------------------- Spectrogram of orignal signal---------------------------------------------------------------#
    with old_spectrogram:
    #Fourier Spectogram Graph 
        fig,ax=plt.subplots(figsize=(8, 3))
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        ax.specgram(original_signal,NFFT=5000,Fs=sample_rate,cmap='jet')
        st.pyplot(fig)
    #-------------------------------- Spectrogram of updated Signal ---------------------------------------------------------------#
    with new_spectrogram:
    #Fourier inverse Spectogram Graph 
        fig,ax=plt.subplots(figsize=(8, 3))
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        ax.specgram(updated_signal,NFFT=5000,Fs=sample_rate,cmap='jet')
        st.pyplot(fig)

if upload_file_flag :
    #----------------------------------------------------Play Audios in Audios Columns---------------------------------------#
    with original_audio_column:
        #Display original audio file
        st.audio(file_uploaded,format='audio/wav')
    with new_audio_slider:
        #Display changed audio file
        if option != "Voice Changer" :
            st.audio("new_audio.wav",format='audio/wav')
        elif option == "Voice Changer":
            st.audio("new_voice.wav",format='audio/wav')
