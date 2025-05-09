import os
from typing import Literal

import gradio as gr
from pip._internal.index import sources

from preprocess_audio import preprocess_audio
from DiarizeFiles import diarize_wav_file
from src.segment_wave_files import segment_wave_files
from src.transcribe_files import transcribe_segments
from src.transcript_analysis import transcript_analysis


def import_audio(file):
    speakers, tmp_file = diarize_wav_file(file)
    speakers = segment_wave_files(speakers, tmp_file)
    os.remove(tmp_file)
    transcript = transcribe_segments(speakers)
    transcript_analysis(transcript)
    summary_output, sentiment_output

    return transcript


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