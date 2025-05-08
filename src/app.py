import streamlit as st
import pandas as pd
from CallCenter import get_included_files, convertMp3ToWav, diarize_wav_file
import os

from segment_wave_files import segment_wave_files
from transcript_analysis import transcript_analysis
from transcribe_files import transcribe_segments

location, wave_files = get_included_files()


def main():
    # --- Streamlit App ---
    st.title("Call Center Analysis")

    # --- Selectable Elements ---

    def get_file_sel():
        if selected_file is None:
            return
        st.session_state.selected_file = location+selected_file  # Store the selected option in session state
        st.write(f"{selected_file} is selected") # Display feedback
        st.session_state.wave_file = st.session_state.selected_file

    selected_file = st.selectbox("Select an element:", wave_files)
    get_file_sel()

    # --- File Upload ---
    uploaded_file = st.file_uploader("Upload a file", type=['mp3', 'wav'])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file.name
        if st.session_state.uploaded_file is not None:
            if os.path.splitext(uploaded_file.name)[1].lower()==".mp3":
                bytes_data = uploaded_file.read()  # read the content of the file in binary
                if not os.path.exists("/tmp"):
                    os.makedirs("/tmp")

                with open(os.path.join("/tmp", uploaded_file.name), "wb") as f:
                    f.write(bytes_data)  # write this content elsewhere
                wav_file = convertMp3ToWav(f"/tmp/{uploaded_file.name}")
                st.session_state.wave_file = wav_file  # Store the selected option in session state
            else:
                bytes_data = uploaded_file.read()  # read the content of the file in binary
                with open(os.path.join("/tmp", uploaded_file.name), "wb") as f:
                    f.write(bytes_data)  # write this content elsewhere
                st.session_state.wave_file = os.path.join("/tmp", uploaded_file.name)  # Store the selected option in session state

            st.write(f"You uploaded: {uploaded_file.name}")  # Display feedback
        uploaded_file = None

    analysis = ""
    # --- Start Button ---
    if st.button("Start"):
        if st.session_state.wave_file is not None:
            with st.spinner("Please wait..."):
                try:
                    speakers = diarize_wav_file(st.session_state.wave_file )

                    speakers = segment_wave_files(speakers, st.session_state.wave_file )

                    transcripts = transcribe_segments(speakers)
                    analysis = transcript_analysis(transcripts)
                except Exception as e:
                    st.error(f"Error processing file: {st.session_state.wave_file}")
        else:
            st.warning("Please upload a file.")

    # --- Reset Button ---
    if st.button("Reset"):
        st.experimental_rerun()

    # --- Large Text Box (Display Only) ---
    if analysis != "":
        analysis = analysis[9:]
        index=analysis.lower().find("sentiment")
        summary=analysis[0:index].lstrip("\n")

        sentiment = analysis[index:]
        index=sentiment.lower().find(":")
        sentiment = sentiment[index+1:].lstrip("\n")

        height = 34*15
        st.text_area("SUMMARY:", value=summary, disabled=True, height=height)
        st.text_area("SENTIMENT:", value=sentiment, disabled=True, height=height)


main()