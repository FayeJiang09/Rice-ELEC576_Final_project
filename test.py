import model.fragmentation_config as fconfig
import model.lstm_tf as lstm
import model.similarity_calc as sim_calc
import model.load_data as load_data
import model.evaluate as evaluate
import numpy as np
import tensorflow as tf
import time
import os

from model.bucket_utils import count_buckets

n = 10000000

#ion_types = ['b{}', 'y{}','b{}-ModLoss','y{}-ModLoss','b{}-NH3','b{}-H2O','y{}-NH3','y{}-H2O']
ion_types = ['b{}', 'y{}','b{}-ModLoss','y{}-ModLoss']
#ion_types = ['b{}','y{}']
mod_config = fconfig.HCD_CommonMod_Config()
mod_config.SetFixMod(['Carbamidomethyl[C]','TMT6plex[AnyN-term]','TMT6plex[K]'])
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
model_folder = './tf-models/train_PT_Hela'
model_name = 'train_PT_Hela-transfer.ckpt'

#ion_types = ['b{}', 'y{}','b{}-ModLoss','y{}-ModLoss','b{}-NH3','b{}-H2O','y{}-NH3','y{}-H2O']
unmod_config = fconfig.HCD_CommonMod_Config()
unmod_config.SetFixMod(['Carbamidomethyl[C]'])
unmod_config.varmod = ["Oxidation[M]","Acetyl[ProteinN-term]"]
unmod_config.SetIonTypes(ion_types)
unmod_config.time_step = 100
unmod_config.min_var_mod_num = 0
unmod_config.max_var_mod_num = 3
#model_folder = './tf-models/train_PT_Hela'
#model_name = 'train_PT_Hela.ckpt'

#unmod_config = fconfig.HCD_Config()
#unmod_config.time_step = 100
#unmod_config.SetIonTypes(ion_types)

#mod_config = fconfig.HCD_CommonMod_Config()
#mod_config.time_step = 100
#mod_config.SetIonTypes(ion_types)
#mod_config.min_var_mod_num = 1
#mod_config.max_var_mod_num = 3

pho_config = fconfig.HCD_pho_Config()
pho_config.time_step = 100
pho_config.SetIonTypes(ion_types)
pho_config.SetFixMod(['Carbamidomethyl[C]','TMT6plex[AnyN-term]','TMT6plex[K]'])
pho_config.varmod = [
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
pho_config.min_var_mod_num = 1
pho_config.max_var_mod_num = 3

colontest = "data/colon_test"
Hela_train = 'data/Hela_train'
Hela_test = 'data/Hela_test'
ProteoToolstrain_25 = 'data/pretrain/train/NCE25'
ProteoToolstrain_30 = 'data/pretrain/train/NCE30'
ProteoToolstrain_35 = 'data/pretrain/train/NCE35'
ProteoToolstest_25 = 'data/pretrain/test/NCE25'
ProteoToolstest_30 = 'data/pretrain/test/NCE30'
ProteoToolstest_35 = 'data/pretrain/test/NCE35'

RNN = lstm.IonLSTM(pho_config)
plot_folder = os.path.join(model_folder, 'log/plots/%s'%model_name)
try:
    os.makedirs(plot_folder)
except:
    pass

pdeep.LoadModel(model_file = os.path.join(model_folder, model_name))

with open(os.path.join(model_folder, 'log/test_%s.txt'%model_name),'w') as log_out:

    def test(folder, ce, ins, n, saveplot, phos = False):
        #print('###################### Begin Unmod ######################', file = log_out)
        #print("[D] " + folder, file = log_out)
        #print("[T] Unmod PSMs:", file = log_out)
        #buckets = load_data.load_folder_as_buckets(folder, unmod_config, nce = ce, instrument = ins, max_n_samples = n)
        #print("[C] " + str(count_buckets(buckets)), file = log_out)
        #output_buckets = pdeep.Predict(buckets)
        #pcc, cos, spc, kdt, SA = sim_calc.CompareRNNPredict_buckets_tf(output_buckets, buckets)
        #sim_names = ['PCC', 'COS', 'SPC', 'KDT', 'SA']
        #print("[A] " + str(evaluate.cum_plot([pcc, cos, spc, kdt, SA], sim_names, saveplot = os.path.join(plot_folder, saveplot+'.eps'), print_file = log_out)), file = log_out)
        #print(folder)
        #print('####################### End Unmod #######################', file = log_out)
        #print("", file = log_out)
        #print("\n", file = log_out)
        #num_pcc_75  = [i for i in pcc if i>0.75]
        #num_pcc_9=[i for i in pcc if i>0.9]
        #print(len(num_pcc_75)/len(pcc),' 75')
        #print(len(num_pcc_9)/len(pcc),' 9')
        #with open('testHela_pcc.txt','w') as filehandle:
        #    for item in pcc:
        #        filehandle.write('%s\n' % item)
        print('####################### Begin Mod #######################', file = log_out)
        print("[D] " + folder, file = log_out)
        if phos: 
            print("[T] Phos PSMs:", file = log_out)
            config = pho_config
            config = mod_config
            mod = '-pho'
        else:
            print("[T] Mod PSMs:", file = log_out)
            config = mod_config
            mod = '-mod'
        buckets = load_data.load_folder_as_buckets(folder, config, nce = ce, instrument = ins, max_n_samples = n)
        print("[C] " + str(count_buckets(buckets)), file = log_out)
        output_buckets = pdeep.Predict(buckets)
        pcc, cos, spc, kdt, SA = sim_calc.CompareRNNPredict_buckets_tf(output_buckets, buckets)
        sim_names = ['PCC', 'COS', 'SPC', 'KDT', 'SA']
        print(pcc,len(pcc),' pcc ',cos,'cos ',spc,'spc ',kdt,'kdt ',SA,'SA ')
        #print("[A] " + str(evaluate.cum_plot([pcc, cos, spc, kdt, SA], sim_names, saveplot = os.path.join(plot_folder, saveplot+mod+'.eps'), print_file = log_out)), file = log_out)
        print('######################## End Mod ########################', file = log_out)
        print("\n", file = log_out)
        num_pcc_75  = [i for i in pcc if i>0.75]
        num_pcc_9=[i for i in pcc if i>0.9]
        print(len(num_pcc_75)/len(pcc),' 75')
        print(len(num_pcc_9)/len(pcc),' 9')
        with open('testcolon_pcc.txt','w') as filehandle:
            for item in pcc:
                filehandle.write('%s\n' % item)
    start_time = time.perf_counter()


    ################### start one folder ##############################
    #test_folder,ce,ins=Hela_test, 0.28,'QE'
    #test(test_folder, ce, ins, n, "Hela_test", phos = False)
    ################### end one folder ################################

    ################### start one folder ##############################
    test_folder,ce,ins = colontest,0.3,'QE'
    test(test_folder, ce, ins, n, "colon", phos = True)
    ################### end one folder ################################
    
    
    ################### start one folder ##############################
    #test_folder,ce,ins = ProteoToolstrain_25,0.25,'Lumos'
    #test(test_folder, ce, ins, n, "ProteoToolstrain_25", phos = False)
    ################## end one folder ################################

    ################### start one folder ##############################
    #test_folder,ce,ins = ProteoToolstrain_30,0.3,'Lumos'
    #test(test_folder, ce, ins, n, "ProteoToolstrain_30", phos = False)
    ################### end one folder ################################


        ################### start one folder ##############################
 #   test_folder,ce,ins =ProteoToolstrain_35,0.35,'Lumos'
  #  test(test_folder, ce, ins, n, "ProteoToolstrain_35", phos = False)
    ################### end one folder ################################

        ################### start one folder ##############################
    #test_folder,ce,ins = ProteoToolstest_25,0.25,'Lumos'
    #test(test_folder, ce, ins, n, "ProteoToolstest_25", phos = False)
    ################### end one folder ################################

        ################### start one folder ##############################
    #test_folder,ce,ins = ProteoToolstest_30,0.3,'Lumos'
    #test(test_folder, ce, ins, n, "ProteoTooltest_30", phos = False)
    ################### end one folder ################################

        ################### start one folder ##############################
    #test_folder,ce,ins = ProteoToolstest_35,0.35,'Lumos'
    #test(test_folder, ce, ins, n, "ProteoToolstest_35", phos = False)
    ################### end one folder ################################



    end_time = time.perf_counter()

    print("time = {:.3f}s".format(end_time - start_time))

pdeep.close_session()
