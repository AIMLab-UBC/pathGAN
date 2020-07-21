from enum import Enum


class OVCAREEnum(Enum):
    CC = 0
    LGSC = 1
    EC = 2
    MC = 3
    HGSC = 4


class TCGAEnum(Enum):
    GLIOMA = 0
    LIVER = 1
    LUNG = 2
    THYROID = 3
    KIDNEY = 4