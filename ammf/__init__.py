import os


def root_dir():
    return os.path.dirname(os.path.realpath(__file__))


def top_dir():
    ammf_root_dir = root_dir()
    return os.path.split(ammf_root_dir)[0]