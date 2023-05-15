import subprocess
import os.path


def check_folder_exist(folder):
    """Check folder exists.

    Parameters
    ----------
    folder : str
        Folder string.
    """
    return os.path.isdir(folder)


def create_folder(root, path=None):
    """Creates a folder with the name 'root' either in the current folder if path is None or a specified path.

    Parameters
    ----------
    root : str
        The name of the created folder.
    path : str, optional
        The name of the path of the created folder.
    """
    if path is None:
        if os.path.isdir(root) is False:
            subprocess.call('mkdir ' + root, shell=True)
    else:
        if os.path.isdir(path+root) is False:
            subprocess.call('mkdir ' + path + root, shell=True)
