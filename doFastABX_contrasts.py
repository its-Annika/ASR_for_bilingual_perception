from fastabx import Dataset
from fastabx import Task
from fastabx import Subsampler, Task
from fastabx import Score
import os
from pathlib import Path
import librosa
import torch
import polars as pl

#if you're running with a GPU,
#CUDA_VISIBLE_DEVICES="" python3 doFastABX_contrasts.py

workingDir = os.getcwd()
itemFile = os.path.join(workingDir, "fastABX_materials", "triPhones.item")
aE = os.path.join(workingDir, "fastABX_materials", "contrastSets", "aE_sampled.item")
aO = os.path.join(workingDir, "fastABX_materials", "contrastSets", "aO_sampled.item")
eE = os.path.join(workingDir, "fastABX_materials", "contrastSets", "eE_sampled.item")
ie = os.path.join(workingDir, "fastABX_materials", "contrastSets", "ie_sampled.item")
oO = os.path.join(workingDir, "fastABX_materials","contrastSets", "oO_sampled.item")
uo = os.path.join(workingDir, "fastABX_materials", "contrastSets", "uo_sampled.item")

modelReps = os.path.join(workingDir, "fastABX_materials")
resultsDir = os.path.join(workingDir, "fastABX_materials", "results")

#stt_ca-es_conformer_transducer_large
#stt_es_conformer_ctc_large

def runTest(MODEL, itemFile, taskType):

     item, features, frequency = itemFile, os.path.join(modelReps, MODEL), 25
     dataset = Dataset.from_item(item, features, frequency)

     task = Task(
          dataset,
          on="#phone",
          across=["speaker"]
          )

     print(f'task length: {len(task)}')

     score = Score(task, "angular")
     details = score.details(levels=["speaker"])
     abx_error_rate = score.collapse(levels=["speaker"])
     breakdown = details.group_by("#phone").agg(pl.col("score").mean()).sort("#phone")

     with open(os.path.join(resultsDir, MODEL+"results_contrasts.txt"), "a+") as f:
          breakdown.write_csv(f)
          f.write(f"Average: {abx_error_rate}\n")
          f.write("\n")


modelsToEval = ["stt_ca-es_conformer_transducer_large", "stt_es_conformer_ctc_large"]

for MODEL in modelsToEval:

     print(f"Evaluating {MODEL}.")
     runTest(MODEL, aE, "aE")
     runTest(MODEL, aO, "aO")
     runTest(MODEL, eE, "eE")
     runTest(MODEL, ie, "ie")
     runTest(MODEL, oO, "oO")
     runTest(MODEL, uo, "uo")


     