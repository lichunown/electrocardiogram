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
        arrythmia_dict = dict(zip(arrythmia, range(len(arrythmia))))

    
    def to_array(data_list:list):
        # input like this: ['57', 'MALE', '窦性心律', '一度房室传导阻滞', 'QRS低电压', '临界ECG']
        #                  ['None', 'None', '窦性心律', 'QRS低电压', '临界ECG']
        result = np.zeros(len(arrythmia))
        if len(data_list) >= 2:
            for item in data_list[2:]:
                result[arrythmia_dict[item]] = 1
        else:
            data_list.append('None') # in test set it can be "[None,]"         
        age = data_list[0] if data_list[0] != 'None' else -1 # age
        if data_list[1] == 'MALE':
            m_or_f = 0
        elif data_list[1] == 'FEMALE':
            m_or_f = 1
        else:
            m_or_f = 0.5
            
        return result, age, m_or_f
    
    
    with open(label_path, 'r', encoding='utf8') as f:
        label = [line.replace('\t\t', '\tNone\t').replace('\t\t', '\tNone\t').split() for line in f]
        # {'1221.txt': ['57', 'MALE', '窦性心律', '一度房室传导阻滞', 'QRS低电压', '临界ECG'],}
        label_file_dir = dict(zip([item[0] for item in label],
                                  [to_array(item[1:]) for item in label]))
    
    # load datas
    all_files_nums = len(os.listdir(os.path.join(base_data_dir, x_dir)))
    all_label = np.zeros([all_files_nums, len(arrythmia)])
    sig_data = np.zeros([all_files_nums, 5000, 8])
    other_data = np.zeros([all_files_nums, 2])
    for nums, filename in enumerate(os.listdir(os.path.join(base_data_dir, x_dir))):
        path = os.path.join(base_data_dir, x_dir, filename)
        with open(path, 'r') as f:
            f.readline()
            per_data = []
            for line in f:
                per_data.append([float(i) for i in line.split()])
        sig_data[nums, :, ] = per_data
        y, age, m_or_f = label_file_dir[filename]
        other_data[nums] = [age, m_or_f]
        all_label[nums] = y
        sys.stdout.write(f'\rprocess in `{x_dir}`: {nums}/{all_files_nums}')
    
    all_data = {
            'label': all_label,
            'sig': sig_data,
            'other': other_data,
            'label_info':arrythmia_dict,
            'other_info': {0: "age", 1: "m_or_f: 0 is male"}
        }
    
#    np.save(os.path.join(base_data_dir, save_prefix + 'data.npy'), all_data)
#    np.save(os.path.join(base_data_dir, save_prefix + 'label.npy'), all_data)
    
    with open(os.path.join(base_data_dir, save_prefix + 'data.pkl'), 'wb') as f:
        pk.dump(all_data, f)
    
    with open(os.path.join(base_data_dir, save_prefix + 'save_info.txt'), 'w') as f:
        f.write('\n'.join([f'{id_}: {name}' for name, id_ in sorted(list(arrythmia_dict.items()), key= lambda x:x[1])]))
        
    return all_data

if __name__ == "__main__":
    _ = pre_process('data/', 'train', 'hf_round1_label.txt', 'hf_round1_arrythmia.txt', 'train_')
    _ = pre_process('data/', 'testA', 'hf_round1_subA.txt', 'hf_round1_arrythmia.txt', 'eval_')


