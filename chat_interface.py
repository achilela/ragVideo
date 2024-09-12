import streamlit as st
from utilsmlm import get_default_rag_chain, split_video

class ChatInterface:
    def __init__(self, config):
        self.config = config
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'mm_rag_chain' not in st.session_state:
            st.session_state.mm_rag_chain = get_default_rag_chain()
        if 'path_to_img' not in st.session_state:
            st.session_state.path_to_img = None
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

    def process_user_input(self, user_input):
        if not st.session_state.path_to_img:
            # First query, need to do RAG
            prompt = self.get_prompt_for_rag()
            executor = st.session_state.mm_rag_chain
            response = executor.invoke(prompt)
            message = response['final_text_output']
            
            if 'metadata' in response['input_to_lvlm']:
                metadata = response['input_to_lvlm']['metadata']
                st.session_state.path_to_img = response['input_to_lvlm'].get('image')
                
                if 'video_path' in metadata:
                    video_path = metadata['video_path']
                    mid_time_ms = metadata['mid_time_ms']
                    splited_video_path = split_video(video_path, mid_time_ms, self.config['output_video_path'])
                    st.session_state.path_to_video = splited_video_path
                
                if 'transcript' in metadata:
                    st.session_state.caption = metadata['transcript']
        else:
            # Subsequent queries, no need for Retrieval
            message = self.config['lvlm_inference_function'](user_input)
        
        return message

    def render(self):
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
        sample_queries = self.config['sample_queries']
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
            self.append_message("user", query)
            
            with st.spinner("Processing your query..."):
                try:
                    response = self.process_user_input(query)
                    self.append_message("assistant", response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            
            st.rerun()
