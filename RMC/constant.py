import enum

from .metrics import mtf
from .metrics import nps
from .metrics import zmtf
from .metrics import nmtf

from .metrics import diameter
from .metrics import radius

from .metrics import static
from .metrics import zstatic
from .metrics import nstatic

# 存放所有数据的根目录
DATADIR = r'C:\Users\99563\Desktop\data'


# 所有计算指标
class METRICS(enum.Enum):
    MTF = mtf
    NPS = nps
    ZMTF = zmtf
    NMTF = nmtf
    DIAMETER = diameter
    RADIUS = radius
    STATIC = static
    ZSTATIC = zstatic
    NSTATIC = nstatic
