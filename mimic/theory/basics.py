import numpy as np


def z2a(z):
    """Redshift to scale factor."""
    return 1./(1.+z)


def a2z(a):
    """Scale factor to redshift."""
    return (1./a) - 1.
