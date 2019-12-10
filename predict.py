import model.fragmentation_config as fconfig
import model.load_data as load_data
from model.bucket_utils import write_buckets, write_buckets_mgf
import numpy as np
import tensorflow as tf
import time

import sys

import model.lstm_tf as lstm
model_folder = './tf-models/train_PT_Hela/'
model = 'train_PT_Hela-transfer.ckpt'
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--nce", default=0.30, help="NCE", type=float)
parser.add_argument("-i", "--instrument", default="Lumos", help="instrument")
parser.add_argument("-in", "--input", default="sample_peptide.txt", help="input peptide file")
parser.add_argument("-out", "--output", default="sample_predict.txt", help="output mgf/txt file")
args = parser.parse_args()

print(vars(args))

CE = args.nce
instrument = args.instrument
in_file = args.input
out_file = args.output

ion_types = ['b{}', 'y{}','b{}-ModLoss','y{}-ModLoss']
mod_config = fconfig.HCD_CommonMod_Config()
mod_config.SetFixMod(['Carbamidomethyl[C]','TMT6plex[AnyN-term]','TMT6plex[K]'])
#mod_config.varmod.extend(['Phospho[S]','Phospho[T]','Phospho[Y]'])
mod_config.varmod = [
    "Acetyl[K]",
    "Biotin[K]",
    "Butyryl[K]",
    "Crotonyl[K]",
    "Dimethyl[R]",
    "Dimethyl[R]",
    "Formyl[K]",
    "Glutaryl[K]",
    "Dicarbamidomethyl[K]",
    "Hydroxyisobutyryl[K]",
    "Malonyl[K]",
    "Methyl[K]",
    "Propionyl[K]",
    "Succinyl[K]",
    "Trimethyl[K]",
    "Oxidation[P]",
    "Deamidated[R]",
    "Dimethyl[K]",
    "Methyl[R]",
    "Nitro[Y]",
    "Phospho[Y]",
    "Phospho[S]",
    "Phospho[T]",
]
mod_config.SetIonTypes(ion_types)
mod_config.time_step = 100
mod_config.min_var_mod_num = 0
mod_config.max_var_mod_num = 3

RNN = lstm.IonLSTM(mod_config)

start_time = time.perf_counter()

buckets = load_data.load_peptide_file_as_buckets(in_file, mod_config, nce = CE, instrument = instrument)
read_time = time.perf_counter()

RNN.LoadModel(model_file = model_folder + model)
output_buckets = RNN.Predict(buckets)
predict_time = time.perf_counter()

write_buckets_mgf(out_file, buckets, output_buckets, mod_config)

print('read time = {:.3f}, predict time = {:.3f}'.format(read_time - start_time, predict_time - read_time))

RNN.close_session()
