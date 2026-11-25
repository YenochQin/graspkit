"""Binary layout shared by Suite's indexed mixing reader and writer."""

from struct import Struct

import numpy as np

SIGNATURE = b"GRMIX001"
INCOMPLETE_SIGNATURE = b"GRMIXTMP"
HEADER = Struct("<8s5q64s")
DIRECTORY_RECORD = Struct("<6q")
LEVEL_DTYPE = np.dtype([("root", "<i8"), ("energy", "<f8")])
