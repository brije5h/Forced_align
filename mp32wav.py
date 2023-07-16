from pydub import AudioSegment

def mp3_to_wav(mp3_path, wav_path):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_path)

    # Export the audio as WAV
    audio.export(wav_path, format="wav")

# Path to the MP3 file
mp3_file = "Voice.mp3"

# Path to save the WAV file
wav_file = "voice.wav"

# Convert MP3 to WAV
mp3_to_wav(mp3_file, wav_file)
