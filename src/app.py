from DiarizeFiles import diarize_wav_file
from segment_wave_files import segment_wave_files
from query_knowledge_base import query_knowledge_base
from transcribe_files import transcribe_segments
from transcript_analysis import transcript_analysis
import os

import gradio as gr
from constants import final_query_model, initial_file_location, DB_PATH
import sqlite3

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain

if final_query_model == "openai":
    embedding_model = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0.2)
else:
    embedding_model = model_name = "raaec/Meta-Llama-3.1-8B-Instruct-Summarizer"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        # max_length=1000,
        max_new_tokens=800,
        eos_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(
        pipeline=text_gen_pipeline,
        model_kwargs={
            "temperature": 0.5,  # Lower = more focused and factual
            "top_k": 50,  # Limits selection to top 50 tokens by probability
            "top_p": 0.85,  # Cumulative probability sampling
            "repetition_penalty": 1.2,  # Penalizes repeated phrases
            "max_new_tokens": 300  # Ensures full, rich output
        }
    )

def import_audio(file):
    speakers, tmp_file = diarize_wav_file(file)
    speakers = segment_wave_files(speakers, tmp_file)
    os.remove(tmp_file)
    transcript = transcribe_segments(speakers)
    # summary_output, sentiment_output = transcript_analysis(transcript)
    transcript, summary_output, sentiment_output, sentiment_score = transcript_analysis(transcript)

    # Connect to (or create) the SQLite database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcript_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript TEXT,
        summary_output TEXT,
        sentiment_output TEXT,
        sentiment_score REAL,
        file_name TEXT
    )
    """)

    # Insert the data into the table
    cursor.execute("""
    INSERT INTO transcript_data (transcript, summary_output, sentiment_output, sentiment_score, file_name)
    VALUES (?, ?, ?, ?, ?)
    """, (transcript, summary_output, sentiment_output, sentiment_score, file))

    # Commit changes and close connection
    conn.commit()
    conn.close()

    print("Data stored successfully in SQLite.")

    return transcript, summary_output, f"{sentiment_output} (score: {sentiment_score})"

# === Gradio UI ===
with gr.Blocks() as demo:
    state = gr.State([])
    gr.Markdown("# 🎙️ Call Center Audio Processor")

    with gr.Tab("🤖 Ask Questions") as question:
        question_input = gr.Textbox(label="Ask a question uploaded ")
        answer_output = gr.Textbox(label="Answer", interactive=False)
        ask_btn = gr.Button("Ask", interactive=True)
        summary_output = gr.Textbox(label="Summary", interactive=False)
        sentiment_output = gr.Textbox(label="Sentiment", interactive=False)
        sentiment_score = gr.Number(label="Sentiment Score", interactive=False)
        trans_output = gr.Textbox(label="Transcript", interactive=False)
        file_name =  gr.Textbox(label="Audio File Name", interactive=False)

        ask_btn.click(
            fn=query_knowledge_base,
            inputs=question_input,
            outputs=[answer_output, trans_output, summary_output,
                     sentiment_output, sentiment_score, file_name]
        )

    with gr.Tab("📥 Upload Audio File") as upload:
        audio_input = gr.Audio(type="filepath", label="Upload WAV or MP3 file",
                               sources=["upload"])
        analyze_btn = gr.Button("Transcribe & Analyze")

        analyze_btn.click(
            fn=import_audio,
            inputs=audio_input,
            outputs=[trans_output, summary_output, sentiment_output]
        )
        trans_output = gr.Textbox(label="Transcript", interactive=False)
        summary_output = gr.Textbox(label="Summary", interactive=False)
        sentiment_output = gr.Textbox(label="Sentiment", interactive=False)

    with gr.Tab("🤖 Import Example Audio Files") as init_db:
        import_btn = gr.Button("Begin Import Audio")
        status_box = gr.Textbox("Uploading Example Audio Files", visible=False)
        def import_example_audio():
            for file in os.listdir(initial_file_location):
                import_audio(initial_file_location + file)

        import_btn.click(
            fn=import_example_audio,
            inputs=None,
            outputs=None
        )


demo.launch()