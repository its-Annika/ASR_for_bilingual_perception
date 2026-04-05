#based on https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb

#makes manifests for common voice data

import os
import librosa as lb
import json
import re
from tqdm.auto import tqdm
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from collections import defaultdict

#modified
def prepare_data(dataPath, split, manifestDir, language):

    transcript = os.path.join(dataPath, split+".tsv")
    
    #make a dictionary of paths, transcripts for the given split
    pathTranscriptDict = {}
    with open(transcript, "r") as f:
        next(f) # don't read the header
        for line in f:
            path = line.split("\t")[1]
            transcript = line.split("\t")[3]
            
            #get rid of quotes in transcripts
            transcript = re.sub(r'[\""]', "", transcript)
            #get rid of punctuation, numbers
            transcript = re.sub(r'[".,!?1234567890"]', "", transcript)
            #get rid of double spaces
            transcript = re.sub(r'\s+', " ", transcript)
            pathTranscriptDict[path] = transcript

    manifest_path = os.path.join(manifestDir, language + "_manifest_" + split + ".json")
    
    print(f"Creating manifest: {manifest_path}")
    
    # Process dataset and create manifest
    manifest_entries = []

    fileCount = 0
    for audioPath in pathTranscriptDict.keys():
        audio_data, sample_rate = lb.load(os.path.join(dataPath, "clips", audioPath))
        duration = len(audio_data) / sample_rate
        transcription = pathTranscriptDict[audioPath]
            
        manifest_entry = {
            "audio_filepath": os.path.join(dataPath, "clips", audioPath),
            "duration": duration,
            "text": transcription.lower().strip()
            }
        
        manifest_entries.append(manifest_entry)
        
        fileCount += 1

        if fileCount % 1000 == 0:
            print(f"Files processed: {fileCount}. Percentage Done: {round(fileCount / len(pathTranscriptDict.keys()), 2)}")
    
    # Write manifest file
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Created manifest with {len(manifest_entries)} entries")
    print(f"Manifest saved to: {manifest_path}")
    
    return str(manifest_path)

def write_processed_manifest(data, original_path):
    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", "_processed.json")

    manifest_dir = os.path.split(original_path)[0]
    filepath = os.path.join(manifest_dir, new_manifest_name)
    write_manifest(filepath, data)
    print(f"Finished writing manifest: {filepath}")
    return filepath

def get_charset(manifest_data):
    charset = defaultdict(int)
    for row in tqdm(manifest_data, desc="Computing character set"):
        text = row['text']
        for character in text:
            charset[character] += 1
    return charset

def remove_special_characters(data):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\…\{\}\【\】\・\。\『\』\、\ー\〜()–¿–«»/]'
    data["text"] = re.sub(chars_to_ignore_regex, '', data["text"]).lower().strip()
    #extra passes for stuborn dashes
    data["text"] = re.sub(r"[—]", '', data["text"])
    data["text"] = re.sub(r"[―]", '', data["text"])
    return data

def apply_preprocessors(manifest, preprocessors):
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])

    print("Finished processing manifest !")
    return manifest

if __name__ == "__main__":

    #note. for the moment, i'm going to use the dev split as train

    workingDir = os.getcwd()
    manifestDir = os.path.join(workingDir, "manifests")
    catalanCorpusPath = ".../corpora/commonVoice/cv-corpus-24.0-2025-12-05/ca"
    #prepare_data(catalanCorpusPath, "dev", manifestDir, "ca")
    #prepare_data(catalanCorpusPath, "test", manifestDir, "ca")
    
    #maifest paths
    dev_manifest_path = os.path.join(manifestDir, "ca_manifest_dev.json")
    test_manifest_path = os.path.join(manifestDir, "ca_manifest_test.json")

    #figure out the char set
    dev_manifest_data = read_manifest(dev_manifest_path)
    test_manifest_data = read_manifest(test_manifest_path)
    dev_text = [data['text'] for data in dev_manifest_data]
    test_text = [data['text'] for data in test_manifest_data]
    dev_charset = get_charset(dev_manifest_data)
    test_charset = get_charset(test_manifest_data)
    dev_set =  set(dev_charset.keys())
    test_set = set(test_charset.keys())

    #final cleaning
    dev_data = read_manifest(dev_manifest_path)
    test_data = read_manifest(test_manifest_path)
    
    PREPROCESSORS = [remove_special_characters]
    dev_data_processed = apply_preprocessors(dev_data, PREPROCESSORS)
    test_data_processed = apply_preprocessors(test_data, PREPROCESSORS)
    dev_manifest_cleaned = write_processed_manifest(dev_data_processed, dev_manifest_path)
    test_manifest_cleaned = write_processed_manifest(test_data_processed, test_manifest_path)

    #print final dev charset
    dev_manifest_data = read_manifest(dev_manifest_cleaned)
    dev_charset = get_charset(dev_manifest_data)

    #write the charset so it can be used in the model training script
    with open(os.path.join(manifestDir, "dev_char_set.tsv"), "w+") as f:
        for char in dev_charset.keys():
            f.write(char + "\t" + str(dev_charset[char]) + "\n")
    f.close()


    