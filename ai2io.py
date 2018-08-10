from os import path, makedirs
import pickle
import json
import os.path
import yaml
import numpy as np
import gzip
import logging
from io import BytesIO
from PIL import Image
import scipy.io
# imported so we have a level of indirection to go through for swapping out
# the definition of glob, isfile
from glob import glob
from os.path import isfile

logger = logging.getLogger(__name__)

def _byteify(data, ignore_dicts = False):
    """
    unicode to plain string - callback function in json lib
    :param data:
    :param ignore_dicts:
    :return:
    """
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data


def load_pickle_object(file_name, compress=True, compress_level=9):
    data = read(file_name)
    if compress:
        load_object = pickle.loads(gzip.decompress(data))
    else:
        load_object = pickle.loads(data)

    return load_object

def dump_pickle_object(dump_object, file_name, compress=True, compress_level=9):
    data = pickle.dumps(dump_object)
    if compress:
        write(file_name, gzip.compress(data, compresslevel=compress_level))
    else:
        write(file_name, data)


def load_json_object(file_name, compress=False):

    if compress:
        return json.loads(gzip.decompress(read(file_name)).decode('utf8'), object_hook=_byteify)
    else:
        return json.loads(read(file_name, 'r'), object_hook=_byteify)


def dump_json_object(dump_object, file_name, compress=False, indent=4):
    data = json.dumps(dump_object, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=indent)
    if compress:
        write(file_name, gzip.compress(data.encode('utf8')))
    else:
        write(file_name, data, 'w')

def dump_json_object_unsorted(dump_object, file_name, compress=False, indent=4):
    data = json.dumps(dump_object, cls=NumpyAwareJSONEncoder, sort_keys=False, indent=indent)
    if compress:
        write(file_name, gzip.compress(data.encode('utf8')))
    else:
        write(file_name, data, 'w')


def load_mat_object(file_name):
    return scipy.io.loadmat(file_name=file_name)


def load_yaml_object(file_name):
    return yaml.load(read(file_name, 'r'))


def read(file_name, mode='rb'):
    with open(file_name, mode) as f:
        return f.read()

def write(file_name, data, mode='wb'):
    with open(file_name, mode) as f:
        f.write(data)

def strip_paths(file_names):
    outfilenames = []
    for filepath in file_names:
        outfilenames.append(path.split(filepath)[1])
    return outfilenames

def short_name(in_name, max_chars=30, fixed_length=True):
    if len(in_name) <= max_chars:
        if fixed_length:
            return in_name.ljust(max_chars)
        else:
            return in_name
    else:
        return in_name[0:max_chars - 7] + '...' + in_name[-10:]


def imread(file_name, flags=1):
    import cv2
    data = np.fromstring(read(file_name), dtype=np.uint8)
    return cv2.imdecode(data, flags)

def imwrite(file_name, img, params=None):
    import cv2
    base, ext = os.path.splitext(file_name)
    result, data = cv2.imencode(ext, img, params)

    if not result:
        raise Exception("could not encode image for filename %s" % file_name)
    write(file_name, data.tostring())

def create_directories(directories):
    for current_dir in directories:
        makedirs(current_dir, exist_ok=True)


def init_logging(log_format='default', log_level='debug'):
    if log_level == 'debug':
        base_logging_level = logging.DEBUG
    elif log_level == 'info':
        base_logging_level = logging.INFO
    elif log_level == 'warning':
        base_logging_level = logging.WARNING
    else:
        raise TypeError('%s is an incorrect logging type!', log_level)
    if len(logger.handlers) == 0:
        ch = logging.StreamHandler()
        logger.setLevel(base_logging_level)
        ch.setLevel(base_logging_level)
        if log_format == 'default':
            formatter = logging.Formatter(fmt='%(asctime)s: %(levelname)s: %(message)s \t[%(filename)s: %(lineno)d]', datefmt='%m/%d %I:%M:%S')
        elif log_format == 'defaultMilliseconds':
            formatter = logging.Formatter(fmt='%(asctime)s: %(levelname)s: %(message)s \t[%(filename)s: %(lineno)d]')
        else:
            formatter = logging.Formatter(fmt=log_format, datefmt='%m/%d %I:%M:%S')

        ch.setFormatter(formatter)
        logger.addHandler(ch)


def fast_deep_copy(in_obj):
    return pickle.loads(pickle.dumps(in_obj))


def serialize_object(in_obj):
    return pickle.dumps(in_obj)


def read_image(image_string, grayscale=False):
    original_image = Image.open(BytesIO(read(image_string)))
    if grayscale:
        rgb_image = original_image.convert('L')
    else:
        rgb_image = original_image.convert('RGB')
    img = np.array(rgb_image)
    if not grayscale:
        img_size = np.shape(img)
        if len(img_size) == 2:
            img = np.dstack((img, img, img))
    return img

def deserialize_object(obj_str):
    return pickle.loads(obj_str)


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.tolist()
            else:
                return [self.default(obj[i]) for i in range(obj.shape[0])]
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int16):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float16):
            return float(obj)
        elif isinstance(obj, np.uint64):
            return int(obj)
        elif isinstance(obj, np.uint32):
            return int(obj)
        elif isinstance(obj, np.uint16):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
