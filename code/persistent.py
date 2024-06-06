import numpy as np
import os


DATA_FOLDER = './data'

RESULTS_FOLDER = './results'
RESULTS_CONFIG_FILE = 'config.txt'
RESULTS_CONFIG_FILE_LAST_EXECUTION_ID_PARAM = 'last_execution_id'

GRAPHS_FOLDER = 'graphs'


def get_execution_id(db_folder_name:np.str_) -> np.int_:
    results_folder_path = f'{RESULTS_FOLDER}'
    create_folder_if_not_exists(results_folder_path)

    db_folder_path = f'{RESULTS_FOLDER}/{db_folder_name}'
    create_folder_if_not_exists(db_folder_path)

    graphs_folder_path = f'{RESULTS_FOLDER}/{db_folder_name}/{GRAPHS_FOLDER}'
    create_folder_if_not_exists(graphs_folder_path)

    config_file_path = f'{RESULTS_FOLDER}/{db_folder_name}/{RESULTS_CONFIG_FILE}'
    if not file_exists(config_file_path):
        experiment_id = 1
    else:
        file = open(config_file_path, 'r')
        lines = file.readlines()

        for line in lines:
            key, value = line.split(':')
            if key == RESULTS_CONFIG_FILE_LAST_EXECUTION_ID_PARAM:
                experiment_id = np.int_(value) + 1

                break

    update_experiment_id(db_folder_name, experiment_id)

    return experiment_id


def update_experiment_id(db_folder_name:np.str_, id:np.int_) -> None:
    config_file_path = f'{RESULTS_FOLDER}/{db_folder_name}/{RESULTS_CONFIG_FILE}'
    line = f'{RESULTS_CONFIG_FILE_LAST_EXECUTION_ID_PARAM}:{id}'

    with open(config_file_path, 'w') as f:
        f.write(line)
    f.close()


def create_folder_if_not_exists(folder_path:np.str_) -> None:
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def file_exists(file_path:np.str_) -> np.bool_:
    if os.path.isfile(file_path):
        return True
    
    return False
