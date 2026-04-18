import os
import json
import torch
import nemo.collections.asr as nemo_asr
from jiwer import cer, wer
from pathlib import Path
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
import math

workingDir = os.getcwd()
dataPath = os.path.join(workingDir, "manifests", "filtered_test.json")
results = os.path.join(workingDir, "cersAndWers_models.txt")
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# read the manifest
evalData = read_manifest(dataPath)
audio_files, references = [], []
for evalItem in evalData:
    audio_files.append(evalItem["audio_filepath"])
    references.append(evalItem["text"].lower())


def getMetrics(model):
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

        
            return(overall_cer, overall_wer)


#load stt_es_conformer
modelPath = os.path.join(workingDir, "stt_es_conformer_ctc_large", "stt_es_conformer_ctc_large.nemo")
#load the model
model = nemo_asr.models.EncDecCTCModelBPE.restore_from(modelPath)
model = model.to(DEVICE)
model.eval()
es = getMetrics(model)

#repeate with stt_ca-es_conformer
modelPath = os.path.join(workingDir, "stt_ca-es_conformer_transducer_large", "stt_ca-es_conformer_transducer_large.nemo")
#load the model
model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(modelPath)
model = model.to(DEVICE)
model.eval()
caes = getMetrics(model)


with open(results, "w+") as f:
    f.write("model\tCER\tWER\n")
    f.write(f"stt_ca-es_conformer_transducer_large\t{caes[0]}\t{caes[1]}\n")
    f.write(f"stt_es_conformer_ctc_large\t{es[0]}\t{es[1]}\n")
    f.close()
