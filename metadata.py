import json
import os


def set_metadata(file_path, key, val):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f, indent=4)
    with open(file_path, 'r') as f:
        data = json.load(f)
    data[key] = val
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def get_metadata(file_path, key):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data[key]

def delete_metadata(file_path, key):
    with open(file_path, 'r') as f:
        data = json.load(f)
    del data[key]
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
