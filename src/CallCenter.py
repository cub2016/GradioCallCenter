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

def convertMp3ToWav(file) :
    # convert mp3 file to a wav file
    sound = AudioSegment.from_mp3(file)
    # sound.export(output_file, format="wav")

    sample_rate = sound.frame_count() / sound.duration_seconds
    print(sample_rate)
    duration = sound.duration_seconds
    sound = sound.set_frame_rate(16000)
    sound = sound.set_channels(1)
    outFile = os.path.splitext(file)[0]+".wav"
    sound.export(outFile, format="wav")
    return outFile

main()
