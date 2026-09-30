from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import numpy as np

CASE_VALID = 1
CASE_INVALID = 2


@dataclass
class ValidationResult:
    case: int                                   # 1 = valid, 2 = invalid
    valid: bool
    message: str
    reasons: List[str] = field(default_factory=list)   # machine-readable codes
    details: List[str] = field(default_factory=list)   # human-readable explanations
    metrics: Dict[str, Any] = field(default_factory=dict)
    # Decoded page (BGR uint8) ready for the preprocessing pipeline. Not serialised.
    image: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("image", None)
        return d
