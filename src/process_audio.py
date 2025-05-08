from array import array

import numpy as np
import librosa
#import sqlite3
#from scipy import sosfilt
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io
import os

def butter_bandpass(lowcut, highcut, fs, order=5):
    sos = butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
    return sos

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    return sosfilt(sos, data)

def normalize_audio(audio_data):
    peak = np.max(np.abs(audio_data))
    if peak == 0:
        return audio_data
    return audio_data / peak

def process_audio(file):
    audio = AudioSegment.from_file(file)
    audio = audio.set_channels(1).set_frame_rate(16000)

    audio_samples = audio.get_array_of_samples()
    audio_samples = audio_samples[0:len(audio_samples)-len(audio_samples)%2]
    samples = np.array(audio_samples, dtype = np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1))
    sr = audio.frame_rate

    lowcut =300.0
    highcut = 4000.0
    filtered = bandpass_filter(samples, lowcut, highcut, fs=sr)
    normalized = normalize_audio(filtered)

    # Convert back to WAV in memory
    back = normalized*(1 << (8 * audio.sample_width - 1))

    back = back.flatten().astype(np.int16)
    back = array('h', back.tolist())
    processed_audio = AudioSegment(
        back,
        frame_rate=sr,
        sample_width=audio.sample_width,
        channels=1
    )

    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    else:
        if os.path.isfile("./tmp/temp.wav"):
            os.remove("./tmp/temp.wav")


    processed_audio.export("./tmp/temp.wav", format="wav")
    return "./tmp/temp.wav"
 #   buffer = io.BytesIO()
 #   processed_audio.export(buffer, format="wav")
 #   wav_bytes = buffer.getvalue()

    # Store in SQLite DB
#    conn = sqlite3.connect(db_path)
#    cursor = conn.cursor()
#    cursor.execute("""
#        CREATE TABLE IF NOT EXISTS processed_audio (
#            id INTEGER PRIMARY KEY AUTOINCREMENT,
#            filename TEXT,
#            audio_blob BLOB
#        )
#    """)
#    cursor.execute(
#        "INSERT INTO processed_audio (filename, audio_blob) VALUES (?, ?)",
#        (os.path.basename(file), wav_bytes))
#    conn.commit()
#    conn.close()
#    print(f"Stored filtered audio for '{file}' in database '{db_path}'.")
#    return True

