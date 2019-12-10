import model.fragmentation_config as fconfig
import model.lstm_tf as lstm
from model.load_data import load_folder_as_buckets as load_folder
from model.bucket_utils import merge_buckets, print_buckets, count_buckets
import numpy as np
import tensorflow as tf
import os

#ion_types = ['b{}', 'y{}','b{}-ModLoss','b{}-NH3','b{}-H2O','y{}-ModLoss','y{}-NH3','y{}-H2O']
ion_types = ['b{}', 'y{}','b{}-ModLoss','y{}-ModLoss']
mod_config = fconfig.HCD_CommonMod_Config()
mod_config.SetFixMod(['Carbamidomethyl[C]'])
mod_config.varmod = ["Oxidation[M]","Acetyl[ProteinN-term]"]
mod_config.SetIonTypes(ion_types)
mod_config.time_step = 100
mod_config.min_var_mod_num = 0
mod_config.max_var_mod_num = 3

RNN = lstm.IonLSTM(mod_config)

RNN.learning_rate = 0.001
RNN.layer_size = 256
RNN.batch_size = 1024
#RNN.batch_size = 128
RNN.BuildModel(input_size = 98, output_size = mod_config.GetTFOutputSize(), nlayers = 2)

RNN.epochs = 100
n = 100000000

out_folder = 'tf-models/train_PT_Hela'
#model_name = 'example.ckpt' # the model is saved as ckpt file
model_name = 'train_PT_Hela.ckpt'

try:
    os.makedirs(out_folder)
except:
    pass
    
#location = "data/all-plabels_zeng/train/"
#file1 = "Gygi-HEK293-NBT-2015-QE-25/plabel"
#file2 = "Kuster-Human-Nature-2014/rectum-Velos-30/plabel"
#file3 = "Mann-FissionYeast-NM-2014-QE-25/new_plabel"
#file4 = "Mann-MouseBrain-NNeu-2015-QEHF-27/new_plabel"
#file5 = "Olsen-CellSys-2017/HelaPhos-QE-28/plabel"
#file6 = "Pandey-Human-Nature-2014"
#file7 = "zengwenfeng-ProteomeTools"


buckets = {}
PT_NCE25 = "data/pretrain/train/NCE25"
PT_NCE30 = "data/pretrain/train/NCE30"
PT_NCE35 = "data/pretrain/train/NCE35"
Hela = "data/Hela_train"

buckets = merge_buckets(buckets, load_folder(Hela, mod_config, nce = 0.28, instrument = 'QE', max_n_samples = n))

buckets = merge_buckets(buckets, load_folder(PT_NCE25, mod_config, nce = 0.25, instrument = 'Lumos', max_n_samples = n))
buckets = merge_buckets(buckets, load_folder(PT_NCE30, mod_config, nce = 0.30, instrument = 'Lumos', max_n_samples = n))
# you can add more plabel-containing folders here
buckets = merge_buckets(buckets, load_folder(PT_NCE35, mod_config, nce = 0.35, instrument = 'Lumos', max_n_samples = n))
print('[I] train data:')
print_buckets(buckets, print_peplen = False)
buckets_count = count_buckets(buckets)
print(buckets_count)
print(buckets_count["total"])


RNN.TrainModel(buckets, save_as = os.path.join(out_folder, model_name))

RNN.close_session()
