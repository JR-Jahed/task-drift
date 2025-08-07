import os

PROJECT_ROOT = os.path.dirname(__file__)

models = {
    'phi3': 'phi__3__3.8',
    'llama3_8b': 'llama__3__8',
}

ROOT_DIR = {
    'phi3': '/mnt/12EA576EEA574D5B/Activation/phi__3__3.8/test',
    'llama3_8b': '/mnt/12EA576EEA574D5B/Activation/llama__3__8B/test'
}

LAYERS_PER_MODEL = {
    'phi3': [0, 7, 15, 23, 31],
    'llama3_8b': [0, 7, 15, 23, 31],
}
