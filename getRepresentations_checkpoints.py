#given an eval manifest, find model representations for files

#ls | sed 's/\.TextGrid$/.mp3/' > filenames.txt  --- to get a list of the grid we have
# grep -Ff names.txt data.json > filtered.json --- to filter the manifests

import os
import nemo.collections.asr as nemo_asr
from pathlib import Path
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
import librosa
import torch
import numpy as np

#set up the gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}.")

#and paths
workingDir = os.getcwd()
#model path, and checkpoint you're evaluting
modelPath = os.path.join(workingDir, "ca_experiments", "default", "2026-03-15_13-53-54","checkpoints")
#eval data
dataPath = os.path.join(workingDir, "manifests", "filtered_test.json")
evalData = read_manifest(dataPath)

modelsToEval = ["epoch_05_valCER_0.1506", "epoch_09_valCER_0.1275", "epoch_14_valCER_0.1132", "epoch_19_valCER_0.1044"]

for EVAlCHECKPOINT in modelsToEval:
    print(f"Evaluating {EVAlCHECKPOINT}.")

    #output for representations
    outputDir = os.path.join(workingDir, "fastABX_materials", EVAlCHECKPOINT)
    os.makedirs(outputDir, exist_ok=True)

    #load checkpoint
    model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(os.path.join(modelPath, EVAlCHECKPOINT+".ckpt"))
    model = model.to(device)

    #get hidden representations (512 hidden state dim)
    for evalItem in evalData:

        audioName = (evalItem["audio_filepath"]).split("/")[-1].removesuffix(".mp3")
        print(audioName)

        #set sr to 16000 kHz for the model
        audio, sr = librosa.load(evalItem["audio_filepath"], sr=16000)
        audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device) 

        length = torch.tensor([audio.shape[-1]], dtype=torch.long).to(device) 

        with torch.no_grad():
            #link to the preprocessor code
            #https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/audio_preprocessing.py
            # Preprocessor (mel spectrogram)
            processed_signal, processed_signal_length =model.preprocessor(
                input_signal=audio,
                length=length,
                )

            #link to encoder code
            #https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/conformer_encoder.py#L311
            # Encoder forward pass
            encoder_outputs, encoder_output_lengths = model.encoder(
                audio_signal=processed_signal,
                length=processed_signal_length,
                )

            #convert [1, 512, T] to [T, 512]
            frames = encoder_outputs.squeeze(0).T.cpu()

            # check frequency for fastABX
            duration = evalItem["duration"]
            print(f"T={frames.shape[0]}, duration={duration:.3f}s, freq={frames.shape[0]/duration:.2f}Hz")

            #save each hidden rep to a file
            torch.save(frames, os.path.join(outputDir, audioName + ".pt"))

