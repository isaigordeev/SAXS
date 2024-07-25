from generation import Generator
from processing import Processing
import os
import json

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

with open(DEFAULT_CONFIG_PATH) as config:
    generation_config = json.load(config)

if __name__ == '__main__':

    # generator = Generator(**generation_config)

    # generator.generation()
    processing = Processing()


    # processing = Processing('/Users/isaigordeev/Desktop/generated/', 4000, 0)
    processing.process()