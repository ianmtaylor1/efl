# Subpackage for experimental models (in development, interesting but slow,
# or just work that I did that would be a shame to trash but isn't very
# useful)

# These models are working, but VERY slow. Not recommended
from .compoisreg         import COMPoisReg,         COMPoisReg_Prior
from .poisreggc          import PoisRegGC,          PoisRegGC_Prior

# These models give divergences which is worrying.
from .consuljainreg      import ConsulJainReg,      ConsulJainReg_Prior
from .consuljainregefgm  import ConsulJainRegEFGM,  ConsulJainRegEFGM_Prior