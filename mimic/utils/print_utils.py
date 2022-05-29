
def printout(statement, level, verbose):
    """Print out leveled (indented) statements.

    Parameters
    ----------
    statement : str
        Statement to write.
    level : int

    """
    if verbose:
        if level == 0:
            print(statement, flush=True)
        elif level == 1:
            print('-->', statement, flush=True)
        elif level == 2:
            print('----->', statement, flush=True)
        elif level == 3:
            print('-------->', statement, flush=True)
        else:
            arrows = '---'*(level-1) + '-->'
            print(arrows, statement, flush=True)


def printspace(verbose):
    if verbose:
        print('', flush=True)
