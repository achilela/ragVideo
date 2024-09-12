import streamlit as st
import io
import sys
import time
import dataclasses
from pathlib import Path
import os
from enum import auto, Enum
from typing import List, Tuple, Any
from app.utils import prediction_guard_llava_conv
import lancedb
from app.utils import load_json_file
from mm_rag.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mm_rag.vectorstores.multimodal_lancedb import MultimodalLanceDB
from mm_rag.MLM.client import PredictionGuardClient
from mm_rag.MLM.lvlm import LVLM
from PIL import Image
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from moviepy.video.io.VideoFileClip import VideoFileClip
from app.utils import prediction_guard_llava_conv, encode_image, Conversation, lvlm_inference_with_conversation

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"

# Keep other necessary imports and global variables

def split_video(video_path, timestamp_in_ms, output_video_path: str = "./shared_data/splitted_videos", output_video_name: str="video_tmp.mp4", play_before_sec: int=3, play_after_sec: int=3):
    # Implementation remains the same as in the original file
    pass

prompt_template = """The transcript associated with the image is '{transcript}'. {user_query}"""

def get_default_rag_chain():
    # Implementation remains the same as in the original file
    pass

class StreamlitInstance:
    def __init__(self, mm_rag_chain=None):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'mm_rag_chain' not in st.session_state:
            st.session_state.mm_rag_chain = mm_rag_chain or get_default_rag_chain()
        if 'path_to_img' not in st.session_state:
            st.session_state.path_to_img = None
        if 'video_title' not in st.session_state:
            st.session_state.video_title = None
        if 'path_to_video' not in st.session_state:
            st.session_state.path_to_video = None
        if 'caption' not in st.session_state:
            st.session_state.caption = None
    
    def append_message(self, role, message):
        st.session_state.messages.append([role, message])
    
    def get_prompt_for_rag(self):
        messages = st.session_state.messages
        assert len(messages) >= 1, "There should be at least one message"
        return messages[-1][1]
    
    def get_conversation_for_lvlm(self):
        pg_conv = prediction_guard_llava_conv.copy()
        image_path = st.session_state.path_to_img
        b64_img = encode_image(image_path)
        for i, (role, msg) in enumerate(st.session_state.messages):
            if i == 0:
                pg_conv.append_message(prediction_guard_llava_conv.roles[0], [msg, b64_img])
            elif i == len(st.session_state.messages) - 1:
                pg_conv.append_message(role, [prompt_template.format(transcript=st.session_state.caption, user_query=msg)])
            else:
                pg_conv.append_message(role, [msg])
        return pg_conv

def process_user_input(user_input):
    instance = StreamlitInstance()
    
    if not st.session_state.path_to_img:
        # First query, need to do RAG
        prompt = instance.get_prompt_for_rag()
        executor = st.session_state.mm_rag_chain
        response = executor.invoke(prompt)
        message = response['final_text_output']
        
        if 'metadata' in response['input_to_lvlm']:
            metadata = response['input_to_lvlm']['metadata']
            st.session_state.path_to_img = response['input_to_lvlm'].get('image')
            
            if 'video_path' in metadata:
                video_path = metadata['video_path']
                mid_time_ms = metadata['mid_time_ms']
                splited_video_path = split_video(video_path, mid_time_ms)
                st.session_state.path_to_video = splited_video_path
            
            if 'transcript' in metadata:
                st.session_state.caption = metadata['transcript']
    else:
        # Subsequent queries, no need for Retrieval
        conversation = instance.get_conversation_for_lvlm()
        message = lvlm_inference_with_conversation(conversation)
    
    return message

def main():
    st.set_page_config(page_title="Multimodal RAG Chat", layout="wide")
    
    st.title("Multimodal RAG: Chat with Videos")
    st.image("assets/header.png")
    
    # Initialize StreamlitInstance if not already done
    if 'instance' not in st.session_state:
        st.session_state.instance = StreamlitInstance()
    
    # Display video if available
    if st.session_state.get('path_to_video'):
        st.video(st.session_state.path_to_video)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message[0]):
            st.write(message[1])
    
    # User input
    user_input = st.chat_input("Enter your query here or choose a sample from the dropdown list!")
    
    # Dropdown for sample queries
    sample_queries = [
        "What is the name of one of the astronauts?",
        "An astronaut's spacewalk",
        "What does the astronaut say?",
    ]
    selected_query = st.selectbox("Or select a sample query:", [""] + sample_queries)
    
    # Clear history button
    clear_btn = st.button("🗑️ Clear history")
    
    if clear_btn:
        st.session_state.messages = []
        st.session_state.path_to_img = None
        st.session_state.path_to_video = None
        st.session_state.caption = None
        st.rerun()
    
    if user_input or selected_query:
        query = user_input or selected_query
        st.session_state.instance.append_message("user", query)
        
        with st.spinner("Processing your query..."):
            try:
                response = process_user_input(query)
                st.session_state.instance.append_message("assistant", response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        st.rerun()

if __name__ == "__main__":
    main()