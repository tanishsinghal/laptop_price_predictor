import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and data

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Custom CSS for background color
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #4CAF50;  /* Green background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Laptop Predictor")
with st.container():
    # Brand selection
    company = st.selectbox('Brand', df['Company'].unique())

    # Type of laptop
    type = st.selectbox('Type', df['TypeName'].unique())

    # RAM selection
    ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

    # Weight input
    weight = st.number_input('Weight of the Laptop')

    # Touchscreen selection
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

    # IPS selection
    ips = st.selectbox('IPS', ['No', 'Yes'])

    # Screen size slider
    screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)

    # Resolution selection
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

    # CPU selection
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())

    # HDD selection
    hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

    # SSD selection
    ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

    # GPU selection
    gpu = st.selectbox('GPU', df['Gpu brand'].unique())

    # OS selection
    os = st.selectbox('OS', df['os'].unique())
    st.markdown('</div',unsafe_allow_html=True)

    if st.button('Predict Price'):
        # Query preparation
            ppi = None
            touchscreen = 1 if touchscreen == 'Yes' else 0
            ips = 1 if ips == 'Yes' else 0

            X_res = int(resolution.split('x')[0])
            Y_res = int(resolution.split('x')[1])
            ppi = ((X_res*2) + (Y_res*2))*0.5 / screen_size
            # query = np.array([ company, type , ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]) ##company, type,
            query = pd.DataFrame([{
                'Company': company,
                'TypeName': type,
                'Ram': ram,
                'Weight': weight,
                'Touchscreen': touchscreen,
                'Ips': ips,
                'ppi': ppi,
                'Cpu brand': cpu,
                'HDD': hdd,
                'SSD': ssd,
                'Gpu brand': gpu,
                'os': os
            }]) 

            # query = query.reshape(1, -1)
            st.success("The predicted price of this configuration is ₹" + str(int(np.exp(pipe.predict(query)[0]))))