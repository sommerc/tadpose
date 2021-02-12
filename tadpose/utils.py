import os


def outfile(main_path):
    def tmp(fn):
        os.makedirs(os.path.join(main_path, os.path.dirname(fn)), exist_ok=True)
        return os.path.join(main_path, fn)

    return tmp
