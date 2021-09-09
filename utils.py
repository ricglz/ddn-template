"""Module containing utility functions and classes"""
from argparse import ArgumentParser
from os import path
from zipfile import ZipFile
from distutils.util import strtobool as fake_strtobool

from tqdm import tqdm
from constant import PROJECT_NAME

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

strtobool = lambda x: bool(fake_strtobool(x))

class CustomParser(ArgumentParser):
    def add_bool_argument(self, flag: str):
        '''Adds a bool argument based on flags'''
        self.add_argument(flag, type=strtobool, nargs='?', const=True, default=False)

def unzip_file(zip_path, dest_path):
    with ZipFile(zip_path, 'r') as zip_file:
        for member in tqdm(zip_file.infolist(), desc='Extracting '):
            zip_file.extract(member, dest_path)

def get_data_dir():
    """Gets the directory where the data is contained"""
    general_dir = '.'
    data_dir = path.join(general_dir, PROJECT_NAME)
    if IN_COLAB and not path.exists(data_dir):
        drive_path = '/content/gdrive'
        drive.mount(drive_path, force_remount=False)
        zip_file = f'{PROJECT_NAME}.zip'
        zip_path = path.join(drive_path, 'MyDrive', zip_file)
        unzip_file(zip_path, general_dir)
    return data_dir
