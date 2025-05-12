import os

embeddings_model = final_query_model = "openai"
llm_analysis_model = "openai"
#llm_analysis_model = "hugging_face"
transcribe_model = "tiny.en" #'tiny.en', 'tiny', 'base.en',
    # 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 
    # 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'

model_name = "raaec/Meta-Llama-3.1-8B-Instruct-Summarizer"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "TheBloke/Llama-2-7b-Chat-GPTQ"

initial_file_location = os.path.join(".", "data") + os.sep
segments_folder = ".segments"
VECTOR_DB_PATH = "chromadb_index"
import_example_files = False