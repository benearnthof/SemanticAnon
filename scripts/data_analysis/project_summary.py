import json
from pathlib import Path
import os

AUDIO = Path(r"C:\Users\Linda Arnthof\Desktop\code\SemanticAnon\data\audio")
DIARIZED = Path(r"C:\Users\Linda Arnthof\Desktop\code\SemanticAnon\data\diarized")
TRANSCRIPTS = Path(r"C:\Users\Linda Arnthof\Desktop\code\SemanticAnon\data\transcripts")

audiodata = os.listdir(AUDIO)
meta = audiodata[-2]
mp3 = audiodata[-1]

print(meta)
print(mp3)

with open(AUDIO/meta) as f:
    metadata = json.load(f)

print(metadata.keys())

diarized = os.listdir(DIARIZED)
diar = diarized[-1]

with open(DIARIZED/diar) as f:
    diar = json.load(f)

print(diar.keys())
print(diar["audio"])
print(diar["transcript"])
print(diar["segments"][0:4])
print(diar["diarization"][0:4])
print(len(diar["speaker_embeddings"]))
print(len(diar["speaker_embeddings"]["SPEAKER_01"]))
print(diar["speaker_embeddings"]["SPEAKER_01"][0:4])

trans = os.listdir(TRANSCRIPTS)
print(trans[0])
with open(TRANSCRIPTS/trans[0]) as f:
    out = f.readlines()

print(out)


# n_speakers = {}
# for d in diarized:
#     with open(DIARIZED/d, encoding="utf-8") as f:
#         out = json.load(f)
#     n_speakers[d] = len(out["speaker_embeddings"].keys())

# k, v = n_speakers.keys(), n_speakers.values()

# from collections import Counter

# counts = Counter(v)