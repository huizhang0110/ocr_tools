import tensorflow as tf
import os


tfrecord_path = "/home/zhui/svt_test.tfrecord"

save_root = "./Datasets/SVT_ASTER"
save_image_dir = os.path.join(save_root, "images")
save_gt_file = os.path.join(save_root, "gt.txt")
if not os.path.exists(save_image_dir):
    os.makedirs(save_image_dir)

gts = []
count = 0
tfrecord_iter = tf.python_io.tf_record_iterator(tfrecord_path)
for serialized_example in tfrecord_iter:
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    transcript = example.features.feature["image/transcript"].bytes_list.value[0].decode()
    image = example.features.feature["image/encoded"].bytes_list.value[0]

    filename = "images/{}.jpg".format(count)
    image_save_path = os.path.join(save_root, filename)

    gts.append("{} {}".format(filename, transcript))
    print(filename, transcript)

    with open(image_save_path, "wb") as f:
        f.write(image)
    count += 1

with open(save_gt_file, "w") as f:
    f.write("\n".join(gts))

print("count: ", count)
