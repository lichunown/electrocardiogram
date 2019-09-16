import os, sys
import numpy as np
import pickle as pk


def pre_process(base_data_dir, x_dir, label_name, arrythmia_name, save_prefix):
    # data save type: (n * 5000 * 8)
    arrythmia_path = os.path.join(base_data_dir, arrythmia_name)
    label_path = os.path.join(base_data_dir, label_name)
    
    
    
    # load label infomations
    with open(arrythmia_path, 'r', encoding='utf8') as f:
        arrythmia = [line.split()[0] for line in f.readlines()]
        # {'完全性左束支传导阻滞': 53, '融合波': 54}
        arrythmia_dict = dict(zip(arrythmia, range(3, len(arrythmia) + 3)))
    arrythmia_dict['age'] = 0
    arrythmia_dict['MALE'] = 1
    arrythmia_dict['FEMALE'] = 2
    
    def to_array(data_list:list):
        # input like this: ['57', 'MALE', '窦性心律', '一度房室传导阻滞', 'QRS低电压', '临界ECG']
        #                  ['None', 'None', '窦性心律', 'QRS低电压', '临界ECG']
        result = np.zeros(1 + 2 + len(arrythmia)) # 2: male or famale; 1:age 
        result[0] = data_list[0] if data_list[0] != 'None' else -1 # age
        if data_list[1] == 'MALE':
            result[1] = 1
        elif data_list[1] == 'FEMALE':
            result[2] = 1
        else:
            result[[1,2]] = 0.5
            
        for item in data_list[2:]:
            result[arrythmia_dict[item]] = 1
        return result
    
    
    with open(label_path, 'r', encoding='utf8') as f:
        label = [line.replace('\t\t', '\tNone\t').replace('\t\t', '\tNone\t').split() for line in f]
        # {'1221.txt': ['57', 'MALE', '窦性心律', '一度房室传导阻滞', 'QRS低电压', '临界ECG'],}
        label_file_dir = dict(zip([item[0] for item in label],
                                  [to_array(item[1:]) for item in label]))
    
    # load datas
    
    all_data = []
    all_label = []
    all_files_nums = len(os.listdir(os.path.join(base_data_dir, x_dir)))
    all_data = np.zeros([all_files_nums, 5000, 8])
    for nums, filename in enumerate(os.listdir(os.path.join(base_data_dir, x_dir))):
        path = os.path.join(base_data_dir, x_dir, filename)
        with open(path, 'r') as f:
            f.readline()
            per_data = []
            for line in f:
                per_data.append([float(i) for i in line.split()])
        all_data[nums] = per_data
        all_label.append(label_file_dir[filename])
        sys.stdout.write(f'\rprocess in `{x_dir}`: {nums}/{all_files_nums}')
    
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    
    np.save(os.path.join(base_data_dir, save_prefix + 'data.npy'), all_data)
    np.save(os.path.join(base_data_dir, save_prefix + 'label.npy'), all_data)
    
    with open(os.path.join(base_data_dir, save_prefix + 'save_info.txt'), 'w') as f:
        f.write('\n'.join([f'{id_}: {name}' for name, id_ in sorted(list(arrythmia_dict.items()), key= lambda x:x[1])]))
        


if __name__ == "__main__":
    pre_process('data/', 'train', 'hf_round1_label.txt', 'hf_round1_arrythmia.txt', 'train_')
    pre_process('data/', 'testA', 'hf_round1_subA.txt', 'hf_round1_arrythmia.txt', 'eval_')


