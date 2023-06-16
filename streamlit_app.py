import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
import av

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

device = 'cpu'
if not hasattr(st, 'model'):
    st.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # vision processing
        flipped = img[:, ::-1, :]

        # model processing
        im_pil = Image.fromarray(flipped)
        results = st.model(im_pil, size=112)
        bbox_img = np.array(results.render()[0])

        return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")


st.title("Webcam and Video Object Detection")
option = st.selectbox("Choose an option", ("Webcam", "Upload Video"))

if option == "Webcam":
    webrtc_ctx = webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,  # Enable asynchronous processing
    )
else:
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
    if uploaded_file is not None:
        video_bytes = uploaded_file.read()
        st.video(video_bytes)

        # Convert video bytes to numpy array
        nparr = np.frombuffer(video_bytes, np.uint8)
        video = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if video is not None:  # Check if video is not None before processing
            # Process each frame of the video
            output_frames = []
            for frame in video:
                im_pil = Image.fromarray(frame)
                results = st.model(im_pil, size=112)
                bbox_img = np.array(results.render()[0])
                output_frames.append(bbox_img)

            # Convert processed frames to video
            output_video = np.stack(output_frames)

            # Display the processed video
            st.video(output_video)
        else:
            st.write(
                "Error reading the video file. Please make sure it is a valid video file.")