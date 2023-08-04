import requests
import numpy as np
from pydub import AudioSegment
import webrtcvad
import io
import transcribe
import redis
import metrics
import os

r = redis.Redis(host='localhost', port=6379, db=0)

test_id = metrics.start_new_test()

# Constants
# url = 'https://gtusieyatzvotohzvlfy.supabase.co/storage/v1/object/public/tmp-data/WWDC_2023.mp3'  # replace with your mp3 url
url = 'https://gtusieyatzvotohzvlfy.supabase.co/storage/v1/object/public/kl8/20230517-155325_8013_(562)307-6777_Incoming_Auto_2144272011031.mp3'
frame_duration = 0.01  # Frame duration in seconds

# Create a VAD object
vad = webrtcvad.Vad()

# Set its aggressiveness mode
vad.set_mode(1)  # Increased to maximum aggressiveness

# Download the whole file
response = requests.get(url)

# Ensure the request was successful
response.raise_for_status()

# Load the entire audio file
audio = AudioSegment.from_mp3(io.BytesIO(response.content))

# Make sure the frame rate is valid
if audio.frame_rate not in [8000, 16000, 32000, 48000]:
    audio = audio.set_frame_rate(32000)  # Set to 16k Hz as an example

# If stereo, split to mono
if audio.channels == 2:
    audio = audio.split_to_mono()[0]

# Calculate the number of samples in 10 ms
frame_samples = int(audio.frame_rate * frame_duration)

# Calculate the number of frames
n_frames = len(audio) // (frame_duration * 1000)  # Multiply frame_duration by 1000 to get ms
print("Number of frames:", n_frames)
print("Frame samples:", frame_samples)

# Try to create audio folder if it doesn't exist
try:
    os.mkdir('./audio')
except:
    pass

transcript_ids = []
monologue_buffer = []
segment_counter = 0
# Process the audio file in 10 ms chunks
for i in range(int(n_frames)):
    start = round(i * frame_duration * 1000)
    end = round((i + 1) * frame_duration * 1000)
    chunk = audio[start:end]
    # chunk = audio[i*frame_duration*1000:(i+1)*frame_duration*1000]  # Multiply frame_duration by 1000 to get ms

    try:
        # Apply VAD on the audio chunk
        raw_audio = np.array(chunk.get_array_of_samples())
        if len(raw_audio) != frame_samples:
            print(f"Skipped frame {i} due to incorrect size: {len(raw_audio)} instead of {frame_samples}")
            continue  # Skip the last frame if it's not the correct size
        
        is_speech = vad.is_speech(raw_audio.tobytes(), sample_rate=chunk.frame_rate)

        if is_speech:
            monologue_buffer.append(chunk)
        else:
            print('Chunk {} does not contain speech'.format(i), flush=True)
            if len(monologue_buffer) > 250:
                segment_counter += 1
                file_name = "./audio/monologue_{}.mp3".format(segment_counter)
                monologue = AudioSegment.empty()
                for chunk in monologue_buffer:
                    monologue += chunk
                monologue.export(file_name, format="mp3")
                monologue_buffer = []
                upload_url = transcribe.upload_file(file_name)
                transcript = transcribe.create_transcript(upload_url, file_name)
                transcript_ids.append(transcript.get('id'))


    except Exception as e:
        print("Unable to process chunk", i, "Error:", str(e))

# Set last transcript id
r.set('test_id:' + str(test_id) + ':last_job', transcript_ids[-1])

