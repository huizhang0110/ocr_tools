import os
import cv2
from multiprocessing import Process
from threading import Thread, Lock

import numpy as np
import time
import pickle
from tqdm import tqdm


class Dataset:

    def __init__(self, tags_file, cache_file, batch_size=64, max_iter=3000000):
        self._batch_size = batch_size
        self._max_iter = max_iter
        self._count = 0

        self._num_cores = 4
        self._num_every_cores = self._batch_size // self._num_cores

        self._all_image_paths = []
        self._all_groundtruth_texts = []
        self._all_shapes = []  # (w, h)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self._all_image_paths, self._all_groundtruth_texts, self._all_shapes = \
                        pickle.load(f)
        else:
            with open(tags_file, "r") as fo:
                lines = fo.readlines()
                for line in tqdm(lines):
                    image_path, groundtruth_text = line.rstrip("\n").split(" ", 1)
                    if groundtruth_text == "":
                        continue
                    try:
                        image = cv2.imread(image_path, 1)
                        h, w = image.shape[:2]
                        new_h = 32
                        new_w = new_h * w / h
                    except Exception as e:
                        print(e)
                        continue
                    self._all_shapes.append((int(new_w), int(new_h)))
                    self._all_image_paths.append(image_path)
                    self._all_groundtruth_texts.append(groundtruth_text)

            with open(cache_file, "wb") as f:
                pickle.dump((
                    self._all_image_paths, 
                    self._all_groundtruth_texts, 
                    self._all_shapes), f)

        self._max_batch_id = len(self) // self._batch_size
        self.threading_lock = Lock()
        assert self._max_batch_id > 0, "Max Batch id less than 0"
        print("Initial Dataset finished!")

    def __len__(self):
        return len(self._all_image_paths)
    
    def _load_image(self, image_paths, groundtruth_texts, resize_wh):
        for image_path, gt in zip(image_paths, groundtruth_texts):
            image = cv2.imread(image_path, 1)
            image = cv2.resize(image, resize_wh)
            image = image.astype(np.float32)
            image = image / 128.0 - 1

            self.threading_lock.acquire()
            self._batch_images.append(image)
            self._batch_groundtruth_texts.append(gt)
            self.threading_lock.release()

    def _get_batch_data(self, start_idx):
        # process [batch_id, batch_id + batch_size)
        image_paths = self._all_image_paths[start_idx : start_idx + self._batch_size]
        groundtruth_texts = self._all_groundtruth_texts[start_idx:start_idx + self._batch_size]
        resize_wh = self._all_shapes[start_idx + self._batch_size - 1]

        self._batch_images = []
        self._batch_groundtruth_texts = []
        thread_list = []
        for i in range(0, self._batch_size, self._num_every_cores):
            p = Thread(
                target=self._load_image,
                args=(image_paths[i:i + self._num_every_cores], 
                      groundtruth_texts[i: i + self._num_every_cores],
                      resize_wh)
            )
            thread_list.append(p)
        for t in thread_list:
            t.setDaemon(True)
            t.start()
        for t in thread_list:
            t.join()
        
        return np.array(self._batch_images, dtype=np.float32), \
                np.array(self._batch_groundtruth_texts)

    def random_get_batch(self):
        start_idx = np.random.randint(0, len(self) - self._batch_size)
        return _get_batch_data(start_idx)

    def data_generator(self):
        while self._count < self._max_iter:
            batch_id = self._count % self._max_batch_id
            start_idx = batch_id * self._batch_size
            yield self._get_batch_data(start_idx)
            self._count += 1
        

if __name__ == "__main__":
    mydataset = Dataset(
        tags_file="/home/zhui/huawei_data/recognition/rel_images/huawei_en_sorted.tags",
        cache_file="/home/zhui/project/atr/experiments/huawei_en_txt/cache.pkl",
        batch_size=32)
    dataset_iterator = mydataset.data_generator()

    avg_time = 0
    num_test = 1000
    for i in range(num_test):
        begin_time = time.time()
        images, gts = next(dataset_iterator)
        print(images, gts)

        end_time = time.time()
        avg_time += (end_time - begin_time)
    print("avg_time: ", avg_time / num_test)
