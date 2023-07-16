import torch
import torchaudio
from dataclasses import dataclass
import IPython
import matplotlib
import matplotlib.pyplot as plt
import re

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed
torch.random.manual_seed(0)

# Set the matplotlib figure size
matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

# Set the file paths
SPEECH_FILE = r"output.wav"
TRANSCRIPT_FILE = "output.txt"
ALIGNED_AUDIO_FILE = "aligned_audio2.wav"
ALIGNED_TRANSCRIPT_FILE = "aligned_transcript2.txt"

# Load the speech waveform and sample rate
waveform, sample_rate = torchaudio.load(SPEECH_FILE)

# Read the transcript from the text file
with open(TRANSCRIPT_FILE, 'r', encoding='utf-8') as file:
    text = file.read()

# Remove symbols from the transcript
transcript = re.sub(r'\W+', '|', text, flags=re.UNICODE)
transcript = transcript.upper()

# Load the pretrained Wav2Vec2 model and labels
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()

# Generate frame-wise label probability
with torch.inference_mode():
    emissions, _ = model(waveform.to(device))
    emissions = torch.log_softmax(emissions, dim=-1)

emission = emissions[0].cpu().detach()

# Generate alignment probability (trellis)
def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

dictionary = {c: i for i, c in enumerate(labels)}
tokens = [dictionary[c] for c in transcript]
trellis = get_trellis(emission, tokens)

# Backtrack to find the most likely path
@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

def backtrack(trellis, emission, tokens, blank_id=0):
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        path.append(Point(j - 1, t - 1, prob))

        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]

path = backtrack(trellis, emission, tokens)

# Merge labels and segments
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float
    def length(self):
        return self.end - self.start

def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

segments = merge_repeats(path)

# Merge words
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length() for seg in segs) / sum(seg.length() for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


word_segments = merge_words(segments)

# Save the aligned audio and transcript
ratio = waveform.size(1) / (trellis.size(0) - 1)

aligned_segments = []
for word_segment in word_segments:
    start = int(word_segment.start * ratio)
    end = int(word_segment.end * ratio)
    aligned_segment = waveform[:, start:end]
    aligned_segments.append(aligned_segment)

aligned_audio = torch.cat(aligned_segments, dim=1)
torchaudio.save(ALIGNED_AUDIO_FILE, aligned_audio, sample_rate)

aligned_transcript = " ".join([word_segment.label for word_segment in word_segments])
with open(ALIGNED_TRANSCRIPT_FILE, "w") as file:
    file.write(aligned_transcript)



# import torchaudio
# import torchaudio.transforms as transforms

# # Load the original WAV file
# waveform, sample_rate = torchaudio.load(SPEECH_FILE)

# # Resample the waveform if needed
# resample_transform = transforms.Resample(sample_rate, desired_sample_rate)
# waveform = resample_transform(waveform)

# # Perform alignment and obtain the segments and transcript
# # ...

# # Save the aligned audio
# aligned_audio_path = "aligned_audio.wav"
# torchaudio.save(aligned_audio_path, waveform, desired_sample_rate)
# print("Aligned audio saved:", aligned_audio_path)

# # Save the transcript
# transcript_path = "aligned_transcript.txt"
# with open(transcript_path, 'w', encoding='utf-8') as file:
#     file.write(transcript)
# print("Aligned transcript saved:", transcript_path)


import torchaudio
import torchaudio.transforms as transforms

# Load the original WAV file
waveform, sample_rate = torchaudio.load(SPEECH_FILE)

# Resample the waveform if needed
desired_sample_rate = 16000  # Replace with your desired sample rate
resample_transform = transforms.Resample(sample_rate, desired_sample_rate)
waveform = resample_transform(waveform)

# Perform alignment and obtain the segments and transcript
# ...

# Save the aligned audio
aligned_audio_path = "aligned_audio.wav"
torchaudio.save(aligned_audio_path, waveform, desired_sample_rate)
print("Aligned audio saved:", aligned_audio_path)

# Save the transcript
transcript_path = "aligned_transcript.txt"
with open(transcript_path, 'w', encoding='utf-8') as file:
    file.write(transcript)
print("Aligned transcript saved:", transcript_path)

