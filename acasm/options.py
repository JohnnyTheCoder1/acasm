from dataclasses import dataclass

@dataclass
class Options:
    unicode: bool = False  # ASCII by default for screen readers
    digits: int = 15       # default precision for evalf

DEFAULT_OPTIONS = Options()
