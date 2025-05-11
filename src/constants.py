import os

embeddings_model = final_query_model = "openai"
llm_analysis_model = "openai"
transcribe_model = "tiny.en" #"tiny.en", "medium.en", "turbo", "large-v3-turbo"

initial_file_location = os.path.join(".", "data") + os.sep
segments_folder = ".segments"
VECTOR_DB_PATH = "chromadb_index"