import torch

from datasets.configs.triangle_config import get_nearest_point_sampling_config
from datasets.dataset import ShapeNetV2Dataset
from tqdm import tqdm, trange
import shutil

from pathlib import Path
import pickle
import threading
import trimesh
import logging
import time

# add filemode="w" to overwrite
logging.basicConfig(filename="preprocess_sdf.log", level=logging.INFO,  format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


DATA_FILE = 'calculated_data_{}.pkl'
PARTITIONS = 100
THREAD_PARTITIONS = 24

WAIT_ALIVE_TIME = 5

SAMPLE_POINTS_COUNT = 500000 // PARTITIONS
MAX_FILE_SIZE_MB = 25 


logger = logging.getLogger(trimesh.__name__)
logger.setLevel(logging.ERROR)



def check_content(path):
    files = Path(path)
    conditions = [len(tuple(files.rglob('*.obj'))) > 0,
                  len(tuple(files.rglob(DATA_FILE.replace('{}','*')))) == PARTITIONS]
    return all(conditions)


def check_existing_parts_before(path, last_idx):
    files = Path(path)
    last_idx += 1
    for i in range(last_idx):
        if len(tuple(files.rglob(DATA_FILE.format(i)))) == 0:
            return i
    return last_idx

def download_data(input_dir, output_dir, timeout, regenerate=True, include_olny=None):
    
    config = get_sdf_sampling_config(SAMPLE_POINTS_COUNT)


    dataset = ShapeNetV2Dataset(input_dir, config, max_MB_file_size=MAX_FILE_SIZE_MB)

    if include_olny:
        files = list(enumerate(dataset.indexes))
        def filtered(x):
            for include_folder in include_olny:
                if x[1].startswith(include_folder):
                    return True
            return False
        elements = sorted(list(map(lambda x: x[0],filter(filtered, files))))
        
    else:
        elements = [k for k in range(len(dataset))]
    


    def dump_data(index, i):
        path = dataset._get_folder_by_id(i)
        dst = Path(output_dir) / path
        try:
            start_time = time.time()

            value = dataset[i]

            for key in list(value.keys()):
                if isinstance(value[key], torch.Tensor):
                    value[key] = value[key].cpu().numpy()

            with open(Path(dst) / DATA_FILE.format(index), 'wb') as file:
                pickle.dump(value, file)

            end_time = time.time()
            logging.info(f"success processing folder {Path(dst) / DATA_FILE.format(index)} at {index} partition, spend time {end_time - start_time}")
        except Exception as e:
            shutil.rmtree(dst, ignore_errors=True)

            logging.error(f"Error in file {Path(dst) / DATA_FILE.format(index)}")
            logging.error(f"Error {e}")
            

            
    threads = []
    
    for partition in range(PARTITIONS):



        for i in tqdm(elements):


            while True:
                threads = list(filter(lambda x: x[1].isAlive(), threads))
                #print(len(threads))
                if len(threads) < THREAD_PARTITIONS:
                    break
                else:
                    time.sleep(WAIT_ALIVE_TIME)
            #if len(threads) >= THREAD_PARTITIONS:
            #    for dest,thread in threads:
            #        try:
            #            thread.join(timeout=timeout)
            #        except Exception as e:
            #            logging.error(f"error on {dest}")
            #            shutil.rmtree(dest, ignore_errors=True)
            #    threads.clear()

            path = dataset._get_folder_by_id(i)


            src = Path(input_dir) / path
            dst = Path(output_dir) / path


            last_existing_part = check_existing_parts_before(dst, partition)
            logging.info(f"{path} has {last_existing_part} partitions last number")
            if not regenerate and last_existing_part > partition:
                logging.info(f'skip {dst} partition {partition} because last_existing_part is {last_existing_part}')
                continue
            if last_existing_part == 0 or regenerate:
                shutil.rmtree(dst, ignore_errors=True)
            dst.mkdir(parents=True, exist_ok=True)



            thread = threading.Thread(target=dump_data, args=(last_existing_part, i))
            thread.start()

            threads.append((dst,thread))

            

        


if __name__ == '__main__':
    path_to_dataset = '/home/malchul/work/datasets/ShapeNetCore.v2'
    output_path = 'data/new_shape_net_sdf'
    TIME = 6*60 #
    logging.info(f"Split data to {SAMPLE_POINTS_COUNT} points in {PARTITIONS} by using {THREAD_PARTITIONS} thread")
    download_data(path_to_dataset, output_path, TIME, regenerate=False, include_olny=['04379243'])
    

    

