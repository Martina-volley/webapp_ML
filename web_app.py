# using streamlit to build a real-time demo of YOLO model

import streamlit as st
#import os,glob
import pandas as pd
import numpy as np
from PIL import Image

st.title('Beach volleyball analytic app')


# Multicolumn Support
col1, col2 = st.columns(2)
frame1 = Image.open("annotated_frame_1.png")
col1.header("Time_1")
col1.image(frame1, use_column_width=True)
frame2 = Image.open("annotated_frame_2.png")
col2.header("Time_2")
col2.image(frame2, use_column_width=True)


