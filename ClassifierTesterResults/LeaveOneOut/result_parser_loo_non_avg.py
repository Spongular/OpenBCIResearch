#This is a quick and dirty script that reads through the .txt files
#that are generated by individual_tests.py and collates the data into
#Pandas dataframes to be converted to excel sheets.

import pandas as pd
import io
import re
import openpyxl

#These are used for looping.
stim_combos = [('hf', 'movement'), ('lr', 'movement')]
columns = None

classifier_type = 'ml'
if classifier_type == 'nn':
    fpath='NN'
    columns = ['layout', 'batch_no',
               'eegnet_accuracy', 'eegnet_recall', 'eegnet_precision', 'eegnet_f1',
               'eegnet_roc-auc', 'fusion_eegnet_accuracy', 'fusion_eegnet_recall',
               'fusion_eegnet_precision', 'fusion_eegnet_f1', 'fusion_eegnet_roc-auc',
               'deep_convnet_accuracy', 'deep_convnet_recall', 'deep_convnet_precision',
               'deep_convnet_f1', 'deep_convnet_roc-auc', 'shallow_convnet_accuracy',
               'shallow_convnet_recall', 'shallow_convnet_precision', 'shallow_convnet_f1', 'shallow_convnet_roc-auc']
else:
    fpath='ML'
    columns = ['layout', 'batch_size',
               'cspknn_accuracy', 'cspknn_recall', 'cspknn_precision', 'cspknn_f1', 'cspknn_roc-auc',
               'cspsvm_accuracy', 'cspsvm_recall', 'cspsvm_precision', 'cspsvm_f1', 'cspsvm_roc-auc',
               'csplda_accuracy', 'csplda_recall', 'csplda_precision', 'csplda_f1', 'csplda_roc-auc',
               'mdm_accuracy', 'mdm_recall', 'mdm_precision', 'mdm_f1', 'mdm_roc-auc',
               'tslr_accuracy', 'tslr_recall', 'tslr_precision', 'tslr_f1', 'tslr_roc-auc',
               'covcsplda_accuracy','covcsplda_recall', 'covcsplda_precision', 'covcsplda_f1', 'covcsplda_roc-auc',
               'covcsplr_accuracy','covcsplr_recall', 'covcsplr_precision', 'covcsplr_f1', 'covcsplr_roc-auc',
               'fbcspsvm_accuracy','fbcspsvm_recall', 'fbcspsvm_precision', 'fbcspsvm_f1', 'fbcspsvm_roc-auc', ]

#This is the general format for the file names.
#sub = subject number, stim = stimulus type, resp = response type, ch = channel layout
file_name_blank = "loo_{stim}_{resp}_openbci_new_{clf}"
layout = 'motor_cortex'
for stim in stim_combos:
    #Create an empty dataframe with our columns
    data_frame = pd.DataFrame(columns=columns, dtype=str)
    print(str(stim))
    #Open the non-fb file.
    fname = file_name_blank.format(stim=stim[0], resp=stim[1], clf=classifier_type)
    path = "{path}/{fname}.txt".format(path=fpath, fname=fname)
    with io.open(path, 'r') as file:
        contents = file.readlines()

    #Now, we fill the list with our data.
    #This will be a 2d list.
    data_list = []
    cur_row = []
    pat1 = re.compile(r"test_\w* = ")
    pat2 = re.compile(r"test_\w*_std = ")
    bat1 = re.compile(r"Value: ")
    bat2 = re.compile(r":\w*")
    for line in contents:
        #Keep track of batch size here.
        if bat1.search(line) is not None:
            if len(cur_row) > 1:
                data_list.append(cur_row)
                cur_row = []
            batch = line.strip('\n')
            batch = re.sub(bat1, '', batch)
            batch = re.sub(bat2, '', batch)
            cur_row.append(layout)
            cur_row.append(batch)
        #Keep track of the data rows here.
        elif pat1.search(line) is not None and pat2.search(line) is None and len(cur_row) > 0:
            data = line.strip('\n')
            data = re.sub(pat1, '', data)
            cur_row.append(data)
    data_list.append(cur_row)

    # Open the fb file.
    if classifier_type == 'ml':
        fname = fname + '_fb'
        path = "{path}/{fname}.txt".format(path=fpath, fname=fname)
        with io.open(path, 'r') as file:
            contents = file.readlines()

        data_list2 = []
        cur_row = []
        pat1 = re.compile(r"test_\w* = ")
        pat2 = re.compile(r"test_\w*_std = ")
        bat1 = re.compile(r"Value: ")
        bat2 = re.compile(r":\w*")
        for line in contents:
            # Keep track of batch size here.
            if bat1.search(line) is not None:
                if len(cur_row) > 1:
                    data_list2.append(cur_row)
                    cur_row = []
                batch = line.strip('\n')
                batch = re.sub(bat1, '', batch)
                batch = re.sub(bat2, '', batch)
                cur_row.append(layout)
                cur_row.append(batch)
            # Keep track of the data rows here.
            elif pat1.search(line) is not None and pat2.search(line) is None and len(cur_row) > 0:
                data = line.strip('\n')
                data = re.sub(pat1, '', data)
                cur_row.append(data)
        data_list2.append(cur_row)

        #Now, we combine our data lists. We basically want to extend each row of the normal data with the fb data.
        for x in range(0, len(data_list)):
            #We extend it with the sliced form of 2, to remove the batch and layout details frm the second list.
            data_list[x].extend(data_list2[x][2:])


    #And finally we append our rows to our data frame.
    for row in data_list:
        r = pd.Series(row, index=data_frame.columns)
        data_frame = data_frame.append(r, ignore_index=True)

    print(data_frame)
    filename = "{stim}-Parsed_Results_{cls}_nonavg_2.xlsx".format(stim=stim[0]+stim[1], cls=classifier_type)
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    data_frame.to_excel(writer)
    writer.save()


