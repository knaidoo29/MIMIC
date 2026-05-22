import numpy as np


def progress_bar(i, total, lenchar, prefix):
    """Display a simple text progress bar to stdout.

    Parameters
    ----------
    i : int
        Current iteration index.
    total : int
        Total number of iterations.
    lenchar : int
        Total width of the printed progress bar string.
    prefix : str
        Prefix string printed before the bar.
    """
    dx = float(total) / float(lenchar - 2)
    doprint = False

    if i == 1 or i == total:
        doprint = True
    else:
        if int((i - 1) / dx) != int(i / dx):
            doprint = True

    if doprint:
        bar_width = max(0, lenchar - 2)
        filled = int(np.floor(i / dx)) if dx != 0 else bar_width
        if filled < 0:
            filled = 0
        if filled > bar_width:
            filled = bar_width
        bar = '#' * filled + '-' * (bar_width - filled)
        print(f"{prefix}|{bar}|")
