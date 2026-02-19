import os
import yaml

class Config:
    def __init__(self, config_file='configs/default.yaml'):
        self.config_file = config_file
        self.params = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        return self.params.get(key, default)

    def __getitem__(self, key):
        return self.get(key)