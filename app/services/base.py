from dataclasses import dataclass


@dataclass()
class BoundingBox:
    """Data class for indexing spatiotemporal data."""

    #: western boundary
    minx: float
    #: eastern boundary
    maxx: float
    #: southern boundary
    miny: float
    #: northern boundary
    maxy: float
    #: earliest boundary
    mint: float
    #: latest boundary
    maxt: float

    @property
    def width(self):
        return self.maxx - self.minx

    @property
    def height(self):
        return self.maxy - self.miny

    def __repr__(self):
        return f"BoundingBox(minx={self.minx}, maxx={self.maxx}, miny={self.miny}, maxy={self.maxy}, \
            mint={self.mint}, maxt={self.maxt})"

    def __post_init__(self) -> None:
        """Validate the arguments passed to :meth:`__init__`.
        Raises:
            ValueError: if bounding box is invalid
                (minx > maxx, miny > maxy, or mint > maxt)
        .. versionadded:: 0.2
        """
        if self.minx > self.maxx:
            raise ValueError(f"Bounding box is invalid: 'minx={self.minx}' > 'maxx={self.maxx}'")
        if self.miny > self.maxy:
            raise ValueError(f"Bounding box is invalid: 'miny={self.miny}' > 'maxy={self.maxy}'")
        if self.mint > self.maxt:
            raise ValueError(f"Bounding box is invalid: 'mint={self.mint}' > 'maxt={self.maxt}'")
