#https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb
#https://github.com/NVIDIA-NeMo/NeMo/blob/stable/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb

#performs model fine-tuning

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
import os
import subprocess
import logging
import copy
from omegaconf import OmegaConf, open_dict
import torch
import lightning.pytorch as ptl
from nemo.utils import exp_manager
from pytorch_lightning.callbacks import Callback

def recoverCharSet(path):
    charDict = {}
    with open(path) as f:
        for line in f:
            charDict[line.split("\t")[0]] = int(line.split("\t")[1].strip("\n"))
    return charDict

def makeTokenizer(TOKENIZER_TYPE,VOCAB_SIZE, dev_manifest, tokenizer_dir):

    cmd = [
        "python3", "process_asr_text_tokenizer.py",
        f"--manifest={dev_manifest}",
        f"--vocab_size={VOCAB_SIZE}",
        f"--data_root={tokenizer_dir}",
        "--tokenizer=spe",
        f"--spe_type={TOKENIZER_TYPE}",
        "--spe_character_coverage=1.0",
        "--no_lower_case",
        "--log"
        ]
    result = subprocess.run(cmd, check=True, text=True)

    with open(os.path.join(tokenizer_dir,'tokenizer.vocab')) as f:
        tokens = f.readlines()
        num_tokens = len(tokens)
        print("Number of tokens : ", num_tokens)
    
    if num_tokens < VOCAB_SIZE:
        print(
            f"The text in this dataset is too small to construct a tokenizer "
            f"with vocab size = {VOCAB_SIZE}. Current number of tokens = {num_tokens}. "
            f"Please reconstruct the tokenizer with fewer tokens"
        )
            
if __name__ == "__main__":
# starting from sub-word encoding ctc model section
#note. for now im using dev as train

#directories
    workingDir = os.getcwd()
    manifestDir = os.path.join(workingDir, "manifests")
    dev_manifest = os.path.join(workingDir, "manifests", "ca_manifest_dev_processed.json")
    test_manifest = os.path.join(workingDir, "manifests", "ca_manifest_test_processed.json")
    corpusDir = ".../corpora/commonVoice/cv-corpus-24.0-2025-12-05/ca"
    tokenizer_dir = os.path.join(workingDir, "tokenizer", "tokenizer_spe_bpe_v48")
    LANGUAGE = "ca"

#tokenizer
    #recover the dev charset
    dev_charset = recoverCharSet(os.path.join(manifestDir, "dev_char_set.tsv"))

    #make the tokenizer
    TOKENIZER_TYPE = "bpe" #@param ["bpe", "unigram"]
    VOCAB_SIZE = len(dev_charset) + 2
    #makeTokenizer(TOKENIZER_TYPE,VOCAB_SIZE, dev_manifest, tokenizer_dir)
    
#loading & tweaking model
    #load model
    model = nemo_asr.models.ASRModel.from_pretrained("stt_es_conformer_ctc_large", map_location='cuda')

    # save decoder weights to possibly restore them
    pretrained_decoder = model.decoder.state_dict() 

    #change the model vocab
    model.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type="bpe")

    #see if decoder weights can be restored
    if model.decoder.decoder_layers[0].weight.shape == pretrained_decoder['decoder_layers.0.weight'].shape:
        print("Decoder shapes matched - restored weights from pre-trained model")
        model.decoder.load_state_dict(pretrained_decoder)
        logging.info("Decoder shapes matched - restored weights from pre-trained model")
    else:
        print("Decoder shapes did not match - could not restore decoder weights from pre-trained model.")
        logging.info("\nDecoder shapes did not match - could not restore decoder weights from pre-trained model.")

    #decide about freezing the encoder
    freeze_encoder = False #@param ["False", "True"] {type:"raw"}
    freeze_encoder = bool(freeze_encoder)

    if freeze_encoder:
        model.encoder.freeze()
        model.encoder.apply(enable_bn_se)
        logging.info("Model encoder has been frozen")
    else:
        model.encoder.unfreeze()
        logging.info("Model encoder has been un-frozen")
    
    #copy model config
    cfg = copy.deepcopy(model.cfg)

    # Setup new tokenizer
    cfg.tokenizer.dir = tokenizer_dir
    cfg.tokenizer.type = "bpe"

    # Set tokenizer config
    model.cfg.tokenizer = cfg.tokenizer

#dataloaders
    #dataloader
    print(OmegaConf.to_yaml(cfg.train_ds))

    # Setup train, validation, test configs -- using dev for train
    with open_dict(cfg):
        # Train dataset
        cfg.train_ds.manifest_filepath = dev_manifest
        cfg.train_ds.is_tarred = False #added
        cfg.train_ds.sample_rate = 16000 #added -- because common voice is at 22kHz
        cfg.train_ds.batch_size = 32
        cfg.train_ds.num_workers = 8
        cfg.train_ds.pin_memory = True
        cfg.train_ds.use_start_end_token = True
        cfg.train_ds.trim_silence = True
        cfg.train_ds.tarred_audio_filepaths = None #added
        cfg.train_ds.augmentor = None #added

        # Validation dataset
        cfg.validation_ds.manifest_filepath = test_manifest
        cfg.validation_ds.sample_rate = 16000 #added -- because common voice is at 22kHz
        cfg.validation_ds.is_tarred = False #added
        cfg.validation_ds.batch_size = 8
        cfg.validation_ds.num_workers = 8
        cfg.validation_ds.pin_memory = True
        cfg.validation_ds.use_start_end_token = True
        cfg.validation_ds.trim_silence = True
        cfg.validation_ds.is_tarred = False #added
        cfg.validation_ds.tarred_audio_filepaths = None #added
        cfg.validation_ds.augmentor = None #added

        # Test dataset
        cfg.test_ds.manifest_filepath = test_manifest
        cfg.test_ds.sample_rate = 16000 #added -- because common voice is at 22kHz
        cfg.test_ds.is_tarred = False #added
        cfg.test_ds.batch_size = 8
        cfg.test_ds.num_workers = 8
        cfg.test_ds.pin_memory = True
        cfg.test_ds.use_start_end_token = True
        cfg.test_ds.trim_silence = True
        cfg.test_ds.is_tarred = False #added
        cfg.test_ds.tarred_audio_filepaths = None #added
        cfg.test_ds.augmentor = None #added

    model.setup_training_data(cfg.train_ds)
    model.setup_multiple_validation_data(cfg.validation_ds)
    model.setup_multiple_test_data(cfg.test_ds)

#optimizer, augmentation, metrics

    print(OmegaConf.to_yaml(cfg.optim))

    with open_dict(model.cfg.optim):
        model.cfg.optim.lr = 0.025
        model.cfg.optim.weight_decay = 0.001
        model.cfg.optim.sched.warmup_steps = None  # Remove default number of steps of warmup
        model.cfg.optim.sched.warmup_ratio = 0.10  # 10 % warmup
        model.cfg.optim.sched.min_lr = 1e-9
    
    with open_dict(model.cfg.spec_augment):
        model.cfg.spec_augment.freq_masks = 2
        model.cfg.spec_augment.freq_width = 25
        model.cfg.spec_augment.time_masks = 10
        model.cfg.spec_augment.time_width = 0.05

        model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)
    
#set up metrics -- use cer 
    use_cer = True #@param ["False", "True"] {type:"raw"}
    log_prediction = True #@param ["False", "True"] {type:"raw"}

    model.wer.use_cer = use_cer
    model.wer.log_prediction = log_prediction
   
#trainer and experiment manager
    if torch.cuda.is_available():
        accelerator = 'gpu'
    else:
        accelerator = 'gpu'

    EPOCHS = 20  

    trainer = ptl.Trainer(devices=1,
                        accelerator=accelerator,
                        max_epochs=EPOCHS,
                        accumulate_grad_batches=1,
                        enable_checkpointing=False, #changed from true 
                        logger=False,
                        log_every_n_steps=5,
                        check_val_every_n_epoch=1) #changed

    # Setup model with the trainer
    model.set_trainer(trainer)
    # finally, update the model's internal config
    model.cfg = model._cfg

    #experiment manager
    os.environ.pop('NEMO_EXPM_VERSION', None)

    config = exp_manager.ExpManagerConfig(
        exp_dir=f'{LANGUAGE}_experiments/',
        create_checkpoint_callback=True,
        checkpoint_callback_params=exp_manager.CallbackParams(
            monitor="val_wer", #note. actually cer because of lines 181-185
            mode="min",
            always_save_nemo=False,
            save_best_model=False, #changed 
            save_top_k=-1, #added
            filename="epoch_{epoch:02d}_valCER_{val_wer:.4f}", #added
            every_n_epochs=1, #added
            save_last=False, #added
            auto_insert_metric_name=False, #added
        ),
    )
    
    config = OmegaConf.structured(config)
    logdir = exp_manager.exp_manager(trainer, config)

    trainer.fit(model)

    save_path = os.path.join(workingDir, "ca_experiments", f"{LANGUAGE}-model-final.nemo")
    model.save_to(save_path)
    print("Model saved.")
