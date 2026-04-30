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
#CUDA_VISIBLE_DEVICES="" python3 doFastABX.py

workingDir = os.getcwd()
itemFile = os.path.join(workingDir, "fastABX_materials", "triPhones.item")
modelReps = os.path.join(workingDir, "fastABX_materials")
resultsDir = os.path.join(workingDir, "fastABX_materials", "results")

#stt_ca-es_conformer_transducer_large
#stt_es_conformer_ctc_large

modelsToEval = ["stt_es_conformer_ctc_large", "stt_ca-es_conformer_transducer_large"]

for MODEL in modelsToEval:
     
     print(f"Evaluating {MODEL}.")

     item, features, frequency = itemFile, os.path.join(modelReps, MODEL), 25
     dataset = Dataset.from_item(item, features, frequency)

     subsampler = Subsampler(max_size_group=10, max_x_across=5)
     task = Task(
          dataset,
          on="#phone",
          by=["next-phone", "prev-phone"],
          across=["speaker"],
          subsampler=subsampler,
     )

     print(f'sub-sampled task length: {len(task)}')

     score = Score(task, "angular")
     details = score.details(levels=[("prev-phone", "next-phone"), "speaker"])
     breakdown = details.group_by("#phone").agg(pl.col("score").mean()).sort("#phone")

     breakdown.write_csv(os.path.join(resultsDir, MODEL+"results.txt"))