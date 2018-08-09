import json
from os import path
from random import choices

if __name__ == '__main__':
    dir_name = '.'
    image_name = 'annotations_00.json'
    file_path = path.join(dir_name, image_name)
    train_percent = 0.5
    test_percent = 1 - train_percent
    with open(file_path, 'r') as f:
        annotations = json.load(f)
        all_keys = [k for k, v in annotations['objects'].items()]
        train_number = int(train_percent*len(all_keys))
        print("all size {}".format(len(all_keys)))
        print(train_number)
        train_set = choices(all_keys, k=train_number)
        all_set = set(all_keys)
        test_set = all_set - set(train_set)

        test_set = list(test_set)

        print('test: len: {} files: {}'.format(len(test_set), test_set))
        print('train len: {} files: {}'.format(len(train_set), train_set))

        print('intersect {}'.format(set(train_set).intersection(set(test_set))))