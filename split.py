import json
from os import path
import random

if __name__ == '__main__':
    dir_name = '.'
    image_name = 'annotations_00.json'
    output_dir = 'annotations_small'
    output_filename = 'split.json'
    file_path = path.join(dir_name, image_name)
    train_percent = 100 / 150
    print("train percent: {}".format(train_percent))
    test_percent = 1 - train_percent
    print("test percent: {}".format(test_percent))
    with open(file_path, 'r') as f:
        annotations = json.load(f)
        print("num objects: {}".format(len(annotations['objects'])))
        all_keys = [k for k, v in annotations['objects'].items()]
        all_set = set(all_keys)
        train_number = int(train_percent*len(all_set))
        print("all size {}".format(len(all_set)))
        print(train_number)
        train_set = set(random.sample(all_set, train_number))
        test_set = all_set - train_set
        print("all set size {}".format(len(all_set)))
        print("train set size {}".format(len(train_set)))
        print("test set size {}".format(len(test_set)))

        test_set = list(test_set)
        train_set = list(train_set)

        spllit_json = {
            'train': train_set,
            'test': test_set,
            'all': test_set + train_set
        }
        with open(path.join(output_dir, output_filename), 'w') as of:
            json.dump(spllit_json, of, indent=4)
        #print('test: len: {} files: {}'.format(len(test_set), test_set))
        #print('train len: {} files: {}'.format(len(train_set), train_set))

        #print('intersect {}'.format(set(train_set).intersection(set(test_set))))
