import os
print(f'FILE: {__file__}')
path = os.path.abspath(__file__)
print(f'PATH: {path}')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f'BASE_DIR: {BASE_DIR}')
MODEL_FILE = os.path.join(BASE_DIR, "../saved_models/1")
print(f'MODEL FILE: {MODEL_FILE}')
