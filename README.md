### Modeling Bilingual Perceptual Learning with Automatic Speech Recognition Models

This project investigates how automatic speech recognition (ASR) models’ perception of speech sounds changes over
the course of continued training, specifically when the additional training data is of a different language than models’ existing training data. By training Spanish ASR models on Catalan audio data and assessing model perception
throughout said continued training, this work intends to explore whether (1) Spanish ASR models’ perception of
Catalan vowels improves over the course of training, and (2) model perceptual trends align with those expected
given human behavioral data.

---
#### The code can be broken down into stages (which should be run in the following order):

**Finetuning**
- commonVoiceDataPrep.py: prepares Common Voice data for finetuning (results in the content of manifests dir)
- process_asr_text_tokenizer.py: makes the new Catalan tokenizer (results in the content of tokenizer dir)
- ASR_CTC_Language_Finetuning.py: conducts finetuning

**Evaluation**
- align.py: samples and force aligns evaluation data
- getCER_andWER_models.py: finds model CER and WER on evaluation data
- getCER_andWER_checkpoints.py: finds checkpoints' CER and WER on evalutation data
- getRepresentations_checkpoints.py: gets checkpoints' representations of evaluation data
- getRepresentations_models.py: gets model representations of evalutation data
- makeItemFile.py: creates item file for ABX task
- doFastABX.py: runs all speech sound ABX task
- doFastABX_contrasts.py: runs contrast-specific ABX tasks (note that contrast-specific item files are created using samplePhones.sh located in fastABX_materials/contrastSets)
  * materials and results are written to fastABX_materials dir
  





