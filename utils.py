"""Module containing utility functions and classes"""
from argparse import ArgumentParser
from distutils.util import strtobool as fake_strtobool

strtobool = lambda x: bool(fake_strtobool(x))

class CustomParser(ArgumentParser):
    def add_bool_argument(self, flag: str):
        '''Adds a bool argument based on flags'''
        self.add_argument(flag, type=strtobool, nargs='?', const=True, default=False)
