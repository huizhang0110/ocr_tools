import lmdb
from tqdm import tqdm
import six
from PIL import Image
import pickle


cvpr_dataset = "./CVPR2016"
env_cvpr = lmdb.open(cvpr_dataset, readonly=True)
txn_cvpr = env_cvpr.begin()

nips_dataset = "./NIPS2014"
env_nips = lmdb.open(nips_dataset, readonly=True)
txn_nips = env_nips.begin()


def get_lmdb_width_map(txn):
    width_map = {}
    n_sample = int(txn.get(b"num-samples").decode())
    for i in tqdm(range(1, n_sample + 1)):
        image_key = b"image-%09d" % i
        image_buf = txn.get(image_key)
        buf = six.BytesIO()
        buf.write(image_buf)
        buf.seek(0)
        try:
            image = Image.open(buf)
            w, h = image.size
        except:
            print("error image", i)
            continue
        new_h = (32 / h) * w
        width_map[image_key] = int(new_h)
    return width_map



# cvpr_width_map = get_lmdb_width_map(txn_cvpr)
# cvpr_width_ordered_list = sorted(cvpr_width_map.items(), key=lambda item: item[1]) 

# nips_width_map = get_lmdb_width_map(txn_nips)
# nips_width_ordered_list = sorted(nips_width_map.items(), key=lambda item: item[1])

# with open("cache.pkl", "wb") as f:
    # pickle.dump((cvpr_width_ordered_list, nips_width_ordered_list), f)
with open("cache.pkl", "rb") as f:
    cvpr_width_ordered_list, nips_width_ordered_list = pickle.load(f)
print("load from pickle")

# merge_dataset
ALL_DATA = "./ALL_REC_DATA"
all_env = lmdb.open(ALL_DATA, map_size=1099511627776)
all_txn = all_env.begin(write=True)
len_cvpr = len(cvpr_width_ordered_list)
len_nips = len(nips_width_ordered_list)
print("cvpr_len", len_cvpr)
print("nips_len", len_nips)

cnt = 0
p_cvpr = 0
p_nips = 0
while (p_cvpr < len_cvpr) and (p_nips < len_nips):
    try:
        cnt += 1
        if cvpr_width_ordered_list[p_cvpr][1] < nips_width_ordered_list[p_nips][1]:
            p_image_key = cvpr_width_ordered_list[p_cvpr][0]
            p_label_key = p_image_key.decode().replace("image", "label").encode()
            image_buf = txn_cvpr.get(p_image_key)
            label_buf = txn_cvpr.get(p_label_key)
            p_cvpr += 1
        else:
            p_image_key = nips_width_ordered_list[p_nips][0]
            p_label_key = p_image_key.decode().replace("image", "label").encode()
            image_buf = txn_nips.get(p_image_key)
            label_buf = txn_nips.get(p_label_key)
            p_nips += 1
        image_key = b"image-%09d" % cnt
        label_key = b"label-%09d" % cnt
        print("1", image_key, " p_cvpr: ", p_cvpr, " p_nips: ", p_nips)
        all_txn.put(image_key, image_buf)
        all_txn.put(label_key, label_buf)
    except Exception as e:
        print("1: ", e)
        print("1", image_key, " p_cvpr: ", p_cvpr, " p_nips: ", p_nips)
        exit()


try:
    while p_cvpr < len_cvpr:
        cnt += 1
        p_image_key = cvpr_width_ordered_list[p_cvpr][0]
        p_label_key = p_image_key.decode().replace("image", "label").encode()
        image_buf = txn_cvpr.get(p_image_key)
        label_buf = txn_cvpr.get(p_label_key)
        p_cvpr += 1
        image_key = b"image-%09d" % cnt
        label_key = b"label-%09d" % cnt
        all_txn.put(image_key, image_buf)
        all_txn.put(label_key, label_buf)
        print("2", image_key)
except Exception as e:
    print("2: ", image_key, " p_cvpr: ", p_cvpr, " p_nips: ", p_nips)
    print("2: ", e)
    exit()

try:
    while p_nips < len_nips:
        cnt += 1
        p_image_key = nips_width_ordered_list[p_nips][0]
        p_label_key = p_image_key.decode().replace("image", "label").encode()
        image_buf = txn_nips.get(p_image_key)
        label_buf = txn_nips.get(p_label_key)
        p_nips += 1
        image_key = b"image-%09d" % cnt
        label_key = b"label-%09d" % cnt
        all_txn.put(image_key, image_buf)
        all_txn.put(label_key, label_buf)
        print("3", image_key)
except Exception as e:
    print("3: ", image_key, " p_cvpr: ", p_cvpr, " p_nips: ", p_nips)
    print("3: ", e)
    exit()


print("total: ", cnt)
all_txn.put(b"num-samples", str(cnt).encode())
all_txn.commit()


env_cvpr.close()
env_nips.close()
all_env.close()

