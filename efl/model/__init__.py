# These models work and are reasonably fast for what they do.
# Considered the "main suite" of models.
from .symordreg          import SymOrdReg,          SymOrdReg_Prior
from .symordreghti       import SymOrdRegHTI,       SymOrdRegHTI_Prior
from .poisregsimple      import PoisRegSimple,      PoisRegSimple_Prior
from .poisregnumberphile import PoisRegNumberphile, PoisRegNumberphile_Prior
from .consuljainreg      import ConsulJainReg,      ConsulJainReg_Prior
from .poisregefgm        import PoisRegEFGM,        PoisRegEFGM_Prior
from .consuljainregefgm  import ConsulJainRegEFGM,  ConsulJainRegEFGM_Prior

# Subpackage for experimental models (in development, interesting but slow,
# or just work that I did that would be a shame to trash but isn't very
# useful)
from . import experimental
