from __future__ import annotations

from turnkey.config import AttackConfig
from turnkey.schema import Sample

from turnkey.components.attacks.core import (
    Attack,
    available_attacks as available_attacks,
    load_attack as load_attack,
    register_attack as register_attack,
)


@register_attack("none")
def _build_none(_: AttackConfig) -> Attack:
    class _NoneAttack(Attack):
        def apply(self, sample: Sample) -> Sample:
            return sample

    return _NoneAttack()

# Concrete attacks register via import side-effects.
from . import artprompt as _artprompt  # noqa: E402,F401
from . import crescendo as _crescendo  # noqa: E402,F401
from . import deepinception as _deepinception  # noqa: E402,F401
from . import manyshot as _manyshot  # noqa: E402,F401
from . import pair as _pair  # noqa: E402,F401
from . import persona as _persona  # noqa: E402,F401
from . import tap as _tap  # noqa: E402,F401
