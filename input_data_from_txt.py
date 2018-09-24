import os
import cv2
from multiprocessing import Process
from threading import Thread

import numpy as np
import time


class Dataset:

    def __init__(self, tags_file, batch_size=64, max_iter=3000000):
        self._batch_size = batch_size
        self._max_iter = max_iter
        self._count = 0

        self._num_cores = 4
        self._num_every_cores = self._batch_size // self._num_cores

        self._all_image_paths = []
        self._all_groundtruth_texts = []
        self._all_shapes = []  # (w, h)

        with open(tags_file, "r") as fo:
            for line in fo:
                image_path, groundtruth_text = line.rstrip("\n").split(" ", 1)
                try:
                    image = cv2.imread(image_path, 1)
                    h, w = image.shape[:2]
                except Exception as e:
                    print(e)
                    continue
                self._all_shapes.append((int(w), int(h)))
                self._all_image_paths.append(image_path)
                self._all_groundtruth_texts.append(groundtruth_text)

        self._max_batch_id = len(self) // self._batch_size

        assert self._max_batch_id > 0, "Max Batch id less than 0"
        print("Initial Dataset finished!")

    def __len__(self):
        return len(self._all_image_paths)
    
    def _load_image(self, image_paths, resize_wh):
        for image_path in image_paths:
            image = cv2.imread(image_path, 1)
            image = cv2.resize(image, resize_wh)
            self._batch_images.append(image)

    def _get_batch_data(self, start_idx):
        # process [batch_id, batch_id + batch_size)
        image_paths = self._all_image_paths[start_idx : start_idx + self._batch_size]
        self._batch_groundtruth_texts = self._all_groundtruth_texts[start_idx:start_idx + self._batch_size]
        self._batch_images = []
        resize_wh = self._all_shapes[start_idx + self._batch_size - 1]
        for i in range(0, self._batch_size, self._num_every_cores):
            p = Process(
                target=self._load_image,
                args=(image_paths[i:i + self._num_every_cores], resize_wh)
            )
            p.start()
        return self._batch_images, self._batch_groundtruth_texts

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
        tags_file="/home/zhui/text_rec/crnn_huawei/data/huawei_data/val.tags",
        batch_size=32)
    dataset_iterator = mydataset.data_generator()

    avg_time = 0
    num_test = 1000
    for i in range(num_test):
        begin_time = time.time()
        batch = next(dataset_iterator)
        end_time = time.time()
        avg_time += (end_time - begin_time)
    print("avg_time: ", avg_time / num_test)
