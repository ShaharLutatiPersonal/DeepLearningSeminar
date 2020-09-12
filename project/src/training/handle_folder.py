import os
import random
def count_entries(d):
    m = {key:len(d[key]) for key in d}
    return m

def db_assignment_problem_solver(base_path = 'D:/Data/data_short/dev-clean'):
    speaker_id_list = os.listdir(base_path)
    files_dict = {}
    for id in speaker_id_list:
        tmp_path = base_path + '/' + id
        folder_list = os.listdir(tmp_path)
        concat_file = []
        for folder in folder_list:
            temporary_tmp_path = tmp_path + '/' + folder
            file_list = os.listdir(temporary_tmp_path)
            file_list = [temporary_tmp_path + '/' + x for x in file_list if 'flac' in x]
            concat_file += file_list
        files_dict[int(id)] = concat_file
    speaker_id_list = [int(x) for x in speaker_id_list]

    total_count = sum([len(files_dict[key]) for key in files_dict])
    pairs = []
    for first_id in files_dict:
        shrinked_list = speaker_id_list[:]
        shrinked_list.remove(first_id)
        for f in files_dict[first_id]:
            choose_bool = False
            counter = count_entries(files_dict)
            while not choose_bool:
                sec_id = random.choice(shrinked_list)
                if counter[sec_id]>0:
                    choose_bool = True
            f1,f2 = f,random.choice(files_dict[sec_id])
            files_dict[sec_id].remove(f2)
            files_dict[first_id].remove(f1)
            pairs.append([f1,f2])
    return pairs

