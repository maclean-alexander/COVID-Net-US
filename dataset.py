import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

INPUT_SHAPE = (480, 480)
COVID_PERCENT = 0.6
mapping = {'negative': 0, 'positive': 1}
NUM_CLASSES = 2
CLASS_WEIGHTS = [1., 1.2]
# Set the locations of train/validation/test, within the datadir passed
# to BalanceCovidDataset - can be the same, as long as the [split].txt
# files are properly defined
TRAIN_FOLDER = 'all_images'
VALID_FOLDER = 'all_images'
TEST_FOLDER = 'all_images'


def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]


def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]


def process_image_file(filepath, size):
    if isinstance(filepath, bytes):
        filepath = filepath.decode('utf-8')
    img = cv2.imread(filepath)
    img = central_crop(img)
    img = cv2.resize(img, (size, size))
    return img


def random_ratio_resize(img, prob=0.3, delta=0.1):
    if np.random.rand() >= prob:
        return img
    ratio = img.shape[0] / img.shape[1]
    ratio = np.random.uniform(max(ratio - delta, 0.01), ratio + delta)

    if ratio * img.shape[1] <= img.shape[1]:
        size = (int(img.shape[1] * ratio), img.shape[1])
    else:
        size = (img.shape[0], int(img.shape[0] / ratio))

    dh = img.shape[0] - size[1]
    top, bot = dh // 2, dh - dh // 2
    dw = img.shape[1] - size[0]
    left, right = dw // 2, dw - dw // 2

    if size[0] > 480 or size[1] > 480:
        print(img.shape, size, ratio)

    img = cv2.resize(img, size)
    img = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT,
                             (0, 0, 0))

    if img.shape[0] != 480 or img.shape[1] != 480:
        raise ValueError(img.shape, size)
    return img


_augmentation_transform = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1),
    zoom_range=(0.85, 1.15),
    fill_mode='constant',
    cval=0.,
)


def apply_augmentation(img):
    img = random_ratio_resize(img)
    img = _augmentation_transform.random_transform(img)
    return img


def _process_csv_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
    return files


class BalanceCovidDataset(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            data_dir,
            csv_file,
            is_training=True,
            batch_size=16,
            input_shape=(480, 480),
            n_classes=2,
            num_channels=3,
            mapping={
                'negative': 0,
                'positive': 1,
            },
            shuffle=True,
            augmentation=apply_augmentation,
            covid_percent=0.5,
            class_weights=[1., 1.]
    ):
        'Initialization'
        self.datadir = data_dir
        self.dataset = _process_csv_file(csv_file)
        self.is_training = is_training
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.mapping = mapping
        self.shuffle = shuffle
        self.covid_percent = covid_percent
        self.class_weights = class_weights
        self.n = 0
        self.augmentation = augmentation

        datasets = {}
        for key in self.mapping.keys():
            datasets[key] = []

        # dataset CSVs have entries of form:
        # {id} {filename} {label} {source_dataset}
        for line in self.dataset:
            datasets[line.split()[2]].append(line)

        if self.n_classes == 2:
            self.datasets = [
                datasets['negative'], datasets['positive']
            ]
        elif self.n_classes == 3:
            self.datasets = [
                datasets['normal'] + datasets['pneumonia'],
                datasets['COVID-19'],
            ]
        else:
            print('Only binary or 3 class classification currently supported.')
        print(len(self.datasets[0]), len(self.datasets[1]))
        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        batch_x, batch_y, weights = self.__getitem__(self.n)
        #print('got next item')
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end()
            self.n = 0
        #print('returning from next')
        return batch_x, batch_y, weights

    def __len__(self):
        return int(np.ceil(len(self.datasets[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            for v in self.datasets:
                np.random.shuffle(v)

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros(
            (self.batch_size, *self.input_shape,
             self.num_channels)), np.zeros(self.batch_size)

        batch_files = self.datasets[0][idx * self.batch_size:(idx + 1) *
                                       self.batch_size]
        # upsample covid cases
        covid_size = max(int(len(batch_files) * self.covid_percent), 1)
        covid_inds = np.random.choice(np.arange(len(batch_files)),
                                      size=covid_size,
                                      replace=False)
        covid_files = np.random.choice(self.datasets[1],
                                       size=covid_size,
                                       replace=False)
        for i in range(covid_size):
            batch_files[covid_inds[i]] = covid_files[i]
        for i in range(len(batch_files)):
            sample = batch_files[i].split()
            # pop off index 0 if sirm
            if sample[-1] == 'sirm':
                sample.pop(0)

            x = process_image_file(os.path.join(self.datadir, TRAIN_FOLDER, sample[1]),
                                   self.input_shape[0])
            if self.is_training and hasattr(self, 'augmentation'):
                x = self.augmentation(x)
            x = x.astype('float32') / 255.0
            y = self.mapping[sample[2]]
            batch_x[i] = x
            batch_y[i] = y
        class_weights = self.class_weights
        weights = np.take(class_weights, batch_y.astype('int64'))
        return batch_x, batch_y, weights


def parse_function_train(images, labels, weights):
    return {'image': images, 'label': labels, 'sample_weights': weights}


def create_weights(batch_y):
    weights = np.take(CLASS_WEIGHTS, batch_y.argmax().astype('int64')).astype('float32')
    return weights


def parse_function_test(images, labels):
    images = tf.py_func(lambda image: np.expand_dims(process_image_file(image, INPUT_SHAPE[0]), axis=0), [images], tf.uint8)
    images = tf.py_func(lambda image: image.astype('float32') / 255.0, [images], tf.float32)
    images = tf.reshape(images, (480, 480, 3)) #need to reshape in order for tensor to have shape
    labels = tf.cast(labels, dtype='int32')
    weights = tf.py_func(create_weights, [labels], tf.float32)
    weights = tf.reshape(weights, [])
    return {'image': images, 'label': labels, 'sample_weights': weights}


class COVIDxInterface:
    def __init__(self, datadir,
                       train_filepath='train_COVIDUS1_convex.txt',
                       valid_filepath='valid_COVIDUS1_convex.txt',
                       test_filepath='test_COVIDUS1_convex.txt'):
        self.train_filepath = train_filepath
        self.valid_filepath = valid_filepath
        self.test_filepath = test_filepath
        self.datadir = datadir
        self.batch_size = 8
        self.batch_size_test = 1

    def get_train_dataset(self, num_shards=1, shard_index=0):
        csv_train = os.path.join(self.datadir, self.train_filepath)
        with open(csv_train) as f:
            filenames = f.readlines()

        num_data = len(filenames)
        batch_size = self.batch_size

        generator = BalanceCovidDataset(data_dir=self.datadir,
                                        csv_file=csv_train,
                                        batch_size=batch_size,
                                        input_shape=INPUT_SHAPE,
                                        n_classes=NUM_CLASSES,
                                        covid_percent=COVID_PERCENT,
                                        class_weights=CLASS_WEIGHTS)

        dataset = tf.data.Dataset.from_generator(lambda: generator,
                                                 output_types=(tf.float32, tf.int32, tf.float32),
                                                 output_shapes=([batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], 3],
                                                                [batch_size],
                                                                [batch_size]))
        dataset = dataset.map(map_func=parse_function_train)
        dataset = dataset.repeat()
        if num_shards > 1:
            dataset = dataset.shard(num_shards, shard_index)

        return dataset, num_data // num_shards, batch_size
      
    def get_valid_dataset(self):
        ''' Separate function for return validation dataset,
        but of same form as self.get_test_dataset()        
        '''
        return self.get_test_dataset(validation=True)

    def get_test_dataset(self, validation=False):
        if validation:
            filepath = self.valid_filepath
            IMAGE_FOLDER = VALID_FOLDER
        else:
            filepath = self.test_filepath
            IMAGE_FOLDER = TEST_FOLDER
        csv_test = os.path.join(self.datadir, self.valid_filepath)
        with open(csv_test) as f:
            filenames = f.readlines()

        image_label = [line.split() for line in filenames]
        imagepath = []
        labels = []
        for info in image_label:
            if os.path.exists(os.path.join(self.datadir, IMAGE_FOLDER, info[1])):
                imagepath.append(os.path.join(self.datadir, IMAGE_FOLDER, info[1]))
                labels.append(mapping[info[2]])
            else:
                print('NOT FOUND: ', info[1])

        num_data = len(labels)
        batch_size = self.batch_size_test

        dataset = tf.data.Dataset.from_tensor_slices((imagepath, labels))
        dataset = dataset.map(map_func=parse_function_test)
        dataset = dataset.batch(batch_size)

        return dataset, num_data, batch_size
