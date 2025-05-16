from preprocess_audio import preprocess_audio
import time
import torch
import os
from pyannote.audio import Pipeline

hugging_face = os.environ.get("HUGGING_FACE")

# model_name = "pyannote/overlapped-speech-detection"

model_name = "pyannote/speaker-diarization-3.1"
# model_name = 'onnx-community/pyannote-segmentation-3.0'
# model_name = "pyannote/segmentation-3.0"
pipelineDiary = Pipeline.from_pretrained(
    model_name,
    use_auth_token=hugging_face)

pipelineDiary.to(torch.device("cuda"))


def diarize_wav_file(file_name):

    tmp_file = preprocess_audio(file_name)

    print("DIARIZING " + file_name)
    start = time.time()
    diarization = pipelineDiary(tmp_file, num_speakers=2)
    print("Elapsed " + str(time.time() - start))
    # {"waveform": audio_tensor, "sample_rate": sample_rate_tensor})
    speakers = []
    contSpeaker = ""
    dict = None
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if contSpeaker != speaker:
            if dict is not None:
                speakers.append(dict)
            dict = {'speaker': speaker, 'start': round(turn.start, 1),
                    'end': round(turn.end, 1)}
            contSpeaker = speaker
        else:
            dict['end'] = round(turn.end, 1)

    return speakers, tmp_file

