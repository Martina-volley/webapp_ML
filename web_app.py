# using streamlit to build a real-time demo of YOLO model
import streamlit as st
from PIL import Image
import cv2
import pandas as pd
import os,glob
import numpy as np
import torch
import time ,sys
#from streamlit_embedcode import github_gist
#import urllib.request
#import urllib
#import moviepy.editor as moviepy
import copy

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.classes=[0] # to filter out the class we need
#result.pandas().xyxy[0] if want to know dataformat of the result
#original classes based on COCO train2017, 80 classes.
yolo_model.conf=0.5 #yolo_model.iou = 0.45 
margin=5
        

def object_detection_image():
    st.title('Object Detection for Images')
    st.subheader("""
    This object detection project takes in an image and outputs the image with bounding boxes created around the objects in the image
    """)
    file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
    if file!= None:
        #img1 = cv2.imread(file)
        #img2=copy.deepcopy(img1)
        img1 = Image.open(file)
        img2 = np.array(img1)

        st.image(img1, caption = "Uploaded Image")
        my_bar = st.progress(0)
        confThreshold =st.slider('Confidence', 0, 100, 50)
        nmsThreshold= st.slider('Threshold', 0, 100, 20)
        obj_list=[]
        ind_ls=[]
        confi_list =[]
        #pred=yolo_model(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        pred=yolo_model(img2)
        classNames = pred.names
        for (xmin, ymin, xmax,   ymax,  confidence,  clas) in pred.xyxy[0].tolist():
            #print(xmin, ymin, xmax,   ymax,  confidence,  clas)
            #print(img2.shape)
            if confidence > (confThreshold/100):
                w,h=int(xmax-xmin),int(ymax-ymin)
                cv2.rectangle(img2, (int(xmin), int(ymin)), (int(xmin+w+margin),int(ymin+h+margin)), (240, 54 , 230), 2)
                cv2.putText(img2, f'{classNames[int(clas)].upper()} {int(confidence*100)}%',
                          (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 0, 240), 2)
                confi_list.append(int(confidence*100))
                obj_list.append(pred.names[int(clas)].upper())  
            
        df= pd.DataFrame(list(zip(obj_list,confi_list)),columns=['Object Name','Confidence'])
        if st.checkbox("Show Object's list" ): 
            st.write(df)
        if st.checkbox("Show Confidence bar chart" ):
            st.subheader('Bar chart for confidence levels')
            st.bar_chart(df["Confidence"]) 
        st.image(img2, caption='Proccesed Image.')
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        my_bar.progress(100)




def main():
    new_title = '<p style="font-size: 42px;">Welcome to Object Detection App Test!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown("""
    This project was built using Streamlit and OpenCV 
    to demonstrate YOLO Object detection in both videos(pre-recorded)
    and images.
    
    
    This YOLO object Detection project can detect 80 objects(i.e classes)
    in either a video or image. The full list of the classes can be found 
    [here](https://github.com/KaranJagtiani/YOLO-Coco-Dataset-Custom-Classes-Extractor/blob/main/classes.txt)"""
    )
    st.sidebar.title("Select Activity")
    choice  = st.sidebar.selectbox("MODE",("About","Object Detection(Image)","Object Detection(Video)"))
    #["Show Instruction","Landmark identification","Show the #source code", "About"]
    
    if choice == "Object Detection(Image)":
        #st.subheader("Object Detection")
        read_me_0.empty()
        read_me.empty()
        #st.title('Object Detection')
        object_detection_image()
    elif choice == "Object Detection(Video)":
        read_me_0.empty()
        read_me.empty()
        #object_detection_video.has_beenCalled = False
        object_detection_video()
        #if object_detection_video.has_beenCalled:
        try:

            clip = moviepy.VideoFileClip('detected_video.mp4')
            clip.write_videofile("myvideo.mp4")
            st_video = open('myvideo.mp4','rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Detected Video") 
        except OSError:
            ''

    elif choice == "About":
        print()
        

if __name__ == '__main__':
		main()	