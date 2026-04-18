import os
import json
import torch
import nemo.collections.asr as nemo_asr
from jiwer import cer, wer
from pathlib import Path
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
import math

workingDir = os.getcwd()
modelPath = modelPath = os.path.join(workingDir, "ca_experiments", "default", "2026-03-15_13-53-54", "checkpoints")
dataPath = os.path.join(workingDir, "manifests", "filtered_test.json")
results = os.path.join(workingDir, "cersAndWers_checkpoints.txt")
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# read the manifest
evalData = read_manifest(dataPath)
audio_files, references = [], []
for evalItem in evalData:
    audio_files.append(evalItem["audio_filepath"])
    references.append(evalItem["text"].lower())


with open(results, "w+") as f:
    f.write("model\tCER\tWER\n")

    for root, dirs, files in os.walk(modelPath):
        for file in files:
            if ".ckpt" in file:
                
                EVAlCHECKPOINT = file.removesuffix(".ckpt")
                print(f"Evaluating {EVAlCHECKPOINT}.")

                #load the model
                model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(
                    os.path.join(modelPath, EVAlCHECKPOINT + ".ckpt")
                )
                model = model.to(DEVICE)
                model.eval()

                #get model transcriptions
                with torch.no_grad():
                    hypotheses = model.transcribe(
                        audio=audio_files,
                        batch_size=BATCH_SIZE,
                    )

                if isinstance(hypotheses[0], list):
                    hypotheses = [h[0] for h in hypotheses]   # take best beam

                hypotheses = [h.text.lower() for h in hypotheses]  # normalise case

                # calculate the cer and wer
                overall_cer = cer(references, hypotheses)
                overall_wer = wer(references, hypotheses)

                f.write(f"{EVAlCHECKPOINT}\t{overall_cer}\t{overall_wer}\n")
                f.flush()
        
f.close()