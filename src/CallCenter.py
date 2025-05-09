import os

from pprint import pprint
import torch
from pydub import AudioSegment

from segment_wave_files import segment_wave_files
from DiarizeFiles import diarize_wav_file
from transcribe_files import transcribe_segments
from transcript_analysis import transcript_analysis

location = os.path.join(".", "data") + os.sep
#location = os.path.join("workspace", "AICallCenter", "data") + os.sep

def get_included_files():
    files = os.listdir(location)

    return location, files

def main():

    dir_list = os.listdir(location)
    for file in dir_list :
        input_file=location+file
        #input_file='C:\\Users\\jerry\\Downloads\\SampleCallsWave\\Tech Support Help from Call Center Experts1.wav'

        # apply pretrained pipeline
        # Pass the audio tensor and sample rate to the pipeline
        speakers, tmp_file = diarize_wav_file(input_file)

        speakers = segment_wave_files(speakers, tmp_file)
        os.remove(tmp_file)
        transcript = transcribe_segments(speakers)
        print(
            "---------------------------------------------------------------------")
        pprint(transcript)
        print("---------------------------------------------------------------------")

        summary = transcript_analysis(transcript)
        pprint(summary) #.encode('utf-8').decode('utf-8'))
        print("\n\n\n\n\n\n\n")

main()
