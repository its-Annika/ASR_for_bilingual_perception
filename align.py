#export PYTHONPATH=$(pwd)/local_packages:$PYTHONPATH
#https://github.com/tabahi/bournemouth-forced-aligner/tree/5e951e72f70fcb4f36dd436b689eb2067c3cfc3a

from bournemouth_aligner import PhonemeTimestampAligner
import os
from pathlib import Path
import json
import librosa
import torch
import random 

random.seed(42)

TARGETNUM = 5000

workingDir = os.getcwd()
outputDir = os.path.join(workingDir, "catalanAlignments", "grids")
os.makedirs(outputDir, exist_ok=True)
testManifest = os.path.join(workingDir, "manifests", "ca_manifest_test_processed.json")

def read_manifest(manifest_path):
    with open(manifest_path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]
    
testData = read_manifest(testManifest)
random.shuffle(testData)

# 1. Create the aligner — it will automatically download the right model
#    Change "en-us" to your language code, e.g. "de", "fr", "es", "hi"
aligner = PhonemeTimestampAligner(preset="ca", device="cuda")

counter = 0
for item in testData:
    audioPath = item["audio_filepath"]
    gridName = audioPath.split("/")[-1].removesuffix(".mp3")
    transcript = item["text"].strip()

    # # 2. Load your audio file (WAV, MP3, FLAC, etc.)
    audio, _ = librosa.load(audioPath, sr=16000, mono=True)
    audio = torch.tensor(audio).unsqueeze(0)  # shape (1, samples)

    # # 3. Run alignment
    # #    Provide the transcript exactly as spoken in the audio
    result = aligner.process_sentence(transcript, audio)
    
    #dont save bad alignments
    if not result["segments"][0]["coverage_analysis"]["bad_alignment"]:
        #save to textgrid
        aligner.convert_to_textgrid(result, os.path.join(outputDir, gridName+".TextGrid"))

        #take the first 5000 valid transcriptions (randomly)
        counter += 1

        if counter % 100 == 0:
            print(f"{counter} grids created and saved.")
            
        if counter >= TARGETNUM:
            break