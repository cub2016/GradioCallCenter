import os
from pydub import AudioSegment
import shutil


def segment_wave_files(speakers, file):

    folder = ".segments"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)

    audio = AudioSegment.from_file(file, format="wav") #.resample(sample_rate_Hz=8000, sample_width=2, channels=1)

    i=0
    speakers_out = []
    for speaker in speakers:
        # {'speaker': speaker, 'start': round(turn.start, 1), 'end': round(turn.end, 1)}
        start = speaker['start']*1000
        stop = speaker['end']*1000
        clip = audio[start:stop]
        clip_name = folder + os.sep + "clipFor" + speaker['speaker'] + "_"+str(i) + ".wav"
        i+=1
        clip.export(clip_name, format="wav")
        speaker['clipFile'] = clip_name
        speakers_out.append(speaker)

    return speakers_out


