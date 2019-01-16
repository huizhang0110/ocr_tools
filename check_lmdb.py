import lmdb
import six
from PIL import Image


all_dataset = "./ALL_REC_DATA"
all_env = lmdb.open(all_dataset, readonly=True)
all_txn = all_env.begin()
n_sample = int(all_txn.get(b"num-samples").decode())

for i in range(1, n_sample + 1):
    image_key = b"image-%09d" % i
    label_key = b"label-%09d" % i

    image_buf = all_txn.get(image_key)
    buf = six.BytesIO()
    buf.write(image_buf)
    buf.seek(0)
    image = Image.open(buf)
    if i == 1 or i == n_sample:
        image.show()

    label = all_txn.get(label_key)
    print(label)

all_env.close()
