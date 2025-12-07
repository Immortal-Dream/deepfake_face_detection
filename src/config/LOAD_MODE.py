from enum import Enum

'''
the mode of loading images
'''
class LOAD_MODE(Enum):
    ONLY_FAKE = 0
    ONLY_REAL = 1
    ALL = None