import gentle

# Paths to audio and transcript files
audio_path = 'voice.wav'
transcript_path = 'Transcript.txt'

# Load the audio and transcript
with open(transcript_path, 'r') as f:
    transcript_text = f.read()

# Perform forced alignment
resources = gentle.Resources()
with gentle.resampled(audio_path) as wavfile:
    aligner = gentle.ForcedAligner(resources, transcript_text)
    result = aligner.transcribe(wavfile)

# Extract the aligned words and their start/end times
aligned_words = result.words
for word in aligned_words:
    print(f"Word: {word.word}, Start: {word.start}, End: {word.end}")
