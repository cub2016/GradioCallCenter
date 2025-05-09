import os
from typing import Literal

import gradio as gr
from langchain_core.documents import Document
from pip._internal.index import sources

from preprocess_audio import preprocess_audio
from DiarizeFiles import diarize_wav_file
from segment_wave_files import segment_wave_files
from transcribe_files import transcribe_segments
from transcript_analysis import transcript_analysis
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

VECTOR_DB_PATH = "faiss_index"
embedding_model = OpenAIEmbeddings()

# def analyze_sentiment(text):
#     return sentiment_analyzer(text)[0]

def build_vectorstore(docs):
    return FAISS.from_documents(docs, embedding_model)

def save_vectorstore(vs, path=VECTOR_DB_PATH):
    vs.save_local(path)

def load_vectorstore(path=VECTOR_DB_PATH):
    return FAISS.load_local(path, embedding_model)

def import_audio(file):
    speakers, tmp_file = diarize_wav_file(file)
    speakers = segment_wave_files(speakers, tmp_file)
    os.remove(tmp_file)
    transcript = transcribe_segments(speakers)
    summary_output, sentiment_output = transcript_analysis(transcript)

    # return transcript, summary_output, sentiment_output

    metadata = {
        "source": file,
        "summary": summary_output,
        "sentiment": sentiment_output["label"],
        "score": sentiment_output["score"]
    }

    doc = Document(page_content=transcript, metadata=metadata)

    # Build or update vectorstore
    if os.path.exists(VECTOR_DB_PATH):
        vs = load_vectorstore()
        vs.add_documents([doc])
    else:
        vs = build_vectorstore([doc])

    save_vectorstore(vs)

    return transcript, summary_output, f"{metadata['sentiment']} (score: {metadata['score']:.2f})"


# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Call Center Audio Processor")

    with gr.Tab("📥 Upload & Analyze"):
        audio_input = gr.Audio(type="filepath", label="Upload WAV or MP3 file", sources=["upload"])
        trans_output = gr.Textbox(label="Transcript")
        summary_output = gr.Textbox(label="Summary")
        sentiment_output = gr.Textbox(label="Sentiment")
        analyze_btn = gr.Button("Transcribe & Analyze")

        analyze_btn.click(
            fn=import_audio,
            inputs=audio_input,
            outputs=[trans_output, summary_output, sentiment_output]
        )

    with gr.Tab("🤖 Ask Questions"):
        question_input = gr.Textbox(label="Ask a question uploaded ")
        answer_output = gr.Textbox(label="Answer")
        ask_btn = gr.Button("Ask")

        # ask_btn.click(
        #     fn=query_knowledge_base,
        #     inputs=question_input,
        #     outputs=answer_output
        # )

demo.launch()