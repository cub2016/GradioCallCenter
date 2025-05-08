import os

import whisper
import time
def transcribe_segments(speakers):
    print(f"Whisper models {whisper.available_models()}")
    model = whisper.load_model("tiny.en", device="cuda")
    #model = whisper.load_model("medium.en", device="cuda")
    #model = whisper.load_model("turbo", device="cuda")
    # model = whisper.load_model("large-v3-turbo", device="cuda")
    transcripts = []
    input_file = ""
    print("Transcribing ALL segments")
    total_start = time.time()
    for speaker in speakers:
        # {'speaker': speaker, 'start': round(turn.start, 1),
        #  'end': round(turn.end, 1), 'clipFile':clipName}
        input_file = speaker['clipFile']

        print("TRANSCRIBING " + input_file)
        start = time.time()
        transcript = model.transcribe(input_file)
        print("Elapsed " + str(time.time() - start))
        segments = transcript["segments"]
        outText = ""
        for segment in segments:
            outText += segment['text']

        transcripts.append(speaker['speaker']+" : "+outText)
        os.remove(input_file)

    print("Total Elapsed " + str(time.time() - total_start))
    print("input_file "+input_file)
    print(os.sep)
    print("input_file[0:input_file.index(os.sep)] "+input_file[0:input_file.index(os.sep)])
    currdir= input_file[0:input_file.index(os.sep)]
    os.rmdir(currdir)

    return transcripts
