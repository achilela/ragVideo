# MAIN_APP
import streamlit as st
from chat_interface import ChatInterface
from utils import load_config, load_environment_variables

def main():
    load_environment_variables()  # Load environment variables at the start
    
    st.set_page_config(page_title="Multimodal RAG Chat", layout="wide")
    
    config = load_config()
    
    st.title("Multimodal RAG: Chat with Videos")
    st.image(config['header_image_path'])
    
    chat_interface = ChatInterface(config)
    chat_interface.render()

if __name__ == "__main__":
    main()
