import yaml
from pathlib import Path
import io
import time
from moviepy.video.io.VideoFileClip import VideoFileClip
from bridgetower_embeddings import BridgeTowerEmbeddings
from multimodal_lancedb import MultimodalLanceDB
from client import PredictionGuardClient
from lvlm import LVLM
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import lancedb

def load_config():
    with open(Path(__file__).parent.parent / 'config' / 'config.yaml', 'r') as file:
        return yaml.safe_load(file)

def split_video(video_path, timestamp_in_ms, output_video_path: str, output_video_name: str="video_tmp.mp4", play_before_sec: int=3, play_after_sec: int=3):
    timestamp_in_sec = int(timestamp_in_ms / 1000)
    Path(output_video_path).mkdir(parents=True, exist_ok=True)
    output_video = Path(output_video_path) / output_video_name
    with VideoFileClip(video_path) as video:
        duration = video.duration
        start_time = max(timestamp_in_sec - play_before_sec, 0)
        end_time = min(timestamp_in_sec + play_after_sec, duration)
        new = video.subclip(start_time, end_time)
        new.write_videofile(str(output_video), audio_codec='aac')
    return str(output_video)

def get_default_rag_chain():
    config = load_config()
    db = lancedb.connect(config['lancedb_host_file'])
    embedder = BridgeTowerEmbeddings()
    vectorstore = MultimodalLanceDB(uri=config['lancedb_host_file'], embedding=embedder, table_name=config['lancedb_table_name'])
    retriever_module = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 1})
    client = PredictionGuardClient()
    lvlm_inference_module = LVLM(client=client)
    
    def prompt_processing(input):
        retrieved_results, user_query = input['retrieved_results'], input['user_query']
        retrieved_result = retrieved_results[0]
        metadata_retrieved_video_segment = retrieved_result.metadata['metadata']
        transcript = metadata_retrieved_video_segment['transcript']
        frame_path = metadata_retrieved_video_segment['extracted_frame_path']
        return {
            'prompt': config['prompt_template'].format(transcript=transcript, user_query=user_query),
            'image' : frame_path,
            'metadata' : metadata_retrieved_video_segment,
        }
    
    prompt_processing_module = RunnableLambda(prompt_processing)
    
    mm_rag_chain_with_retrieved_image = (
        RunnableParallel({"retrieved_results": retriever_module , 
                          "user_query": RunnablePassthrough()}) 
        | prompt_processing_module
        | RunnableParallel({'final_text_output': lvlm_inference_module, 
                            'input_to_lvlm' : RunnablePassthrough()})
    )
    return mm_rag_chain_with_retrieved_image

# Add other utility functions as needed
