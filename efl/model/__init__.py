# These models work and are reasonably fast for what they do.
# Considered the "main suite" of models.
from .symordreg          import SymOrdReg,          SymOrdReg_Prior
from .symordreghti       import SymOrdRegHTI,       SymOrdRegHTI_Prior
from .poisregnumberphile import PoisRegNumberphile, PoisRegNumberphile_Prior
from .poisregsimple      import PoisRegSimple,      PoisRegSimple_Prior
from .consuljainreg      import ConsulJainReg,      ConsulJainReg_Prior
from .poisregefgm        import PoisRegEFGM,        PoisRegEFGM_Prior

# These models are working, but VERY slow. Not recommended
from .compoisreg         import COMPoisReg,         COMPoisReg_Prior
from .poisreggc          import PoisRegGC,          PoisRegGC_Prior
