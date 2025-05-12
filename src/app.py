from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from DiarizeFiles import diarize_wav_file
from segment_wave_files import segment_wave_files
from transcribe_files import transcribe_segments
from transcript_analysis import transcript_analysis
import os
import re
import gradio as gr
from constants import final_query_model, VECTOR_DB_PATH, initial_file_location, import_example_files

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from langchain.schema import HumanMessage

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

def build_vectorstore(docs, path=VECTOR_DB_PATH):
    return Chroma.from_documents(collection_name="Call_Center_Data", documents=docs,
                                 embedding=embedding_model, persist_directory=path)

def load_vectorstore(path=VECTOR_DB_PATH):
    return Chroma(persist_directory=path, embedding_function=embedding_model)

# === Query Interface ===
def query_knowledge_base(user_question):
    if not os.path.exists(VECTOR_DB_PATH):
        return "No data in vector store. Please upload audio first."

    vs = load_vectorstore()

    retriever = vs.as_retriever() # search_kwargs={"k": 15}

    relevant_docs = retriever.invoke(user_question)
    context = "\n\n".join([doc.metadata["source"]+"   "+doc.page_content  for doc in relevant_docs])

    for doc in relevant_docs:
        print(doc.metadata["score"]+"   "+ doc.metadata["source"])

    template = """
        You are an assistant that answers questions based on the context provided.
        If you don't know the answer, say you don't know. 
        keep the answer concise. 
        Context:
        {context}
    
        Question:
        {question}
    
        Answer in a clear and concise manner.
    """
    prompt = PromptTemplate.from_template(template)

    filled_prompt = prompt.format(context=context, question=user_question)

    response = llm.invoke([HumanMessage(content=filled_prompt)])

    # Regular expression pattern to match .wav or .mp3 file names
    pattern = r'([a-zA-Z0-9_\\]+\.wav|[a-zA-Z0-9_\\]+\.mp3)'

    # Search for the pattern in the input string
    match = re.search(pattern, response.content)

    # Check if a match was found and extract the file name
    if match:
        file_name = match.group(0)
        file_name = file_name[file_name.rfind(os.sep):]
    else:
        print("No file name found.")
        file_name=""

    transcript = ""
    sentiment_analysis = ""
    summary = ""
    sentiment_score = ""

    source_doc = ""
    if file_name != "":
        for doc in relevant_docs:
            file = doc.metadata["source"]
            match = re.search(pattern, doc.metadata["source"])
            if match:
                doc_file = match.group(0)
                doc_file = doc_file[doc_file.rfind(os.sep):]
            else:
                continue
            if doc_file == file_name:
                source_doc = doc
                break

    if source_doc != "":
        transcript = source_doc.page_content
        summary = source_doc.metadata["summary"]
        sentiment_analysis = source_doc.metadata["sentiment"]
        sentiment_score = source_doc.metadata["score3"]

    print("=== Answer ===")
    print(response.content)
    return response.content, transcript, summary, sentiment_analysis, sentiment_score, file_name

def import_audio(file):
    speakers, tmp_file = diarize_wav_file(file)
    speakers = segment_wave_files(speakers, tmp_file)
    os.remove(tmp_file)
    transcript = transcribe_segments(speakers)
    # summary_output, sentiment_output = transcript_analysis(transcript)
    transcript, summary_output, sentiment_output, sentiment_score = transcript_analysis(transcript)

    metadata = {
        "source": file,
        "summary": summary_output,
        "sentiment": sentiment_output,
        "score": sentiment_score
    }

    doc = Document(page_content=transcript, metadata=metadata)

    # Build or update vectorstore
    if os.path.exists(VECTOR_DB_PATH):
        vs = load_vectorstore()
        vs.add_documents([doc])
    else:
        vs = build_vectorstore([doc])

    return transcript, summary_output, f"{metadata['sentiment']} (score: {metadata['score']})"

# === Gradio UI ===
with gr.Blocks() as demo:
    vector_db_initialized = False
    if import_example_files:
        for file in os.listdir(initial_file_location):
             import_audio(initial_file_location + file)

    gr.Markdown("# 🎙️ Call Center Audio Processor")

    with gr.Tab("🤖 Ask Questions"):
        question_input = gr.Textbox(label="Ask a question uploaded ")
        answer_output = gr.Textbox(label="Answer")
        ask_btn = gr.Button("Ask")

        ask_btn.click(
            fn=query_knowledge_base,
            inputs=question_input,
            outputs=answer_output
        )

    with gr.Tab("📥 Call Details"):
        trans_output = gr.Textbox(label="Transcript")
        summary_output = gr.Textbox(label="Summary")
        sentiment_output = gr.Textbox(label="Sentiment")

    with gr.Tab("📥 Upload"):
        audio_input = gr.Audio(type="filepath", label="Upload WAV or MP3 file",
                               sources=["upload"])
        analyze_btn = gr.Button("Transcribe & Analyze")

        analyze_btn.click(
            fn=import_audio,
            inputs=audio_input,
            outputs=[trans_output, summary_output, sentiment_output]
        )

demo.launch()