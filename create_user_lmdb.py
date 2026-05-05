import os
import lmdb
import cv2
from tqdm import tqdm
import numpy as np
import io
from PIL import Image

def get_datalist(data_dir, label_file, max_len=25):
    """
    Load data from txt label file.
    Format: path/to/img\tlabel
    """
    data_list = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc=f'Loading {label_file}'):
            parts = line.strip('\n').split('\t')
            if len(parts) < 2:
                # Try fallback if tab didn't work (though it should)
                parts = line.strip('\n').split(' ')
            
            if len(parts) >= 2:
                img_path = os.path.join(data_dir, parts[0].strip())
                label = parts[1].strip()
                if len(label) > max_len:
                    continue
                if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                    data_list.append([img_path, label])
    return data_list

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(data_list, outputPath, checkValid=True):
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=2 * 1024 * 1024 * 1024) # 2GB
    cache = {}
    cnt = 1
    for imagePath, label in tqdm(data_list, desc=f'Creating LMDB at {outputPath}'):
        try:
            with open(imagePath, 'rb') as f:
                imageBin = f.read()
                buf = io.BytesIO(imageBin)
                with Image.open(buf) as img:
                    w, h = img.size
            
            if checkValid:
                imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
                img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
                if img is None or img.shape[0] * img.shape[1] == 0:
                    continue

            imageKey = 'image-%09d'.encode() % cnt
            labelKey = 'label-%09d'.encode() % cnt
            whKey = 'wh-%09d'.encode() % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label.encode()
            cache[whKey] = (str(w) + '_' + str(h)).encode()

            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
            cnt += 1
        except Exception as e:
            print(f'Error processing {imagePath}: {e}')
            continue

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print(f'Created dataset with {nSamples} samples')

if __name__ == '__main__':
    # Configuration
    DATA_ROOT = r'D:\IEEE\data\phD\data\50k_OCR\aaa_train_ne_version_5'
    TRAIN_LABELS = os.path.join(DATA_ROOT, 'train_labels.txt')
    VAL_LABELS = os.path.join(DATA_ROOT, 'val_labels.txt')
    TEST_LABELS = os.path.join(DATA_ROOT, 'test_labels.txt')
    
    OUTPUT_ROOT = r'D:\IEEE\data\phD\OpenOCR\train_data_lmdb'
    
    # Process Train
    print("Processing Training Set...")
    train_data = get_datalist(DATA_ROOT, TRAIN_LABELS)
    createDataset(train_data, os.path.join(OUTPUT_ROOT, 'train'))
    
    # Process Val
    print("\nProcessing Validation Set...")
    val_data = get_datalist(DATA_ROOT, VAL_LABELS)
    createDataset(val_data, os.path.join(OUTPUT_ROOT, 'val'))

    # Process Test
    print("\nProcessing Test Set...")
    test_data = get_datalist(DATA_ROOT, TEST_LABELS)
    createDataset(test_data, os.path.join(OUTPUT_ROOT, 'test'))
