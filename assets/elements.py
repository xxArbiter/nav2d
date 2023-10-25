from abc import ABC, abstractmethod
from typing import Union, List, Literal
from enum import Enum
import numpy as np


class Element(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __repr__(self) -> str:
        ...


class Point(Element):
    def __init__(self, x: float, y: float) -> None:
        self._x = x
        self._y = y
        self._pos = np.array([x, y])

    @property
    def pos(self) -> np.array:
        return self._pos

    def __add__(self, vector: "Vector") -> "Point":
        if not isinstance(vector, Vector):
            raise TypeError("Can only add a vector to a point.")
        return Point(*(self._pos + vector._pos))

    def __sub__(self, other: "Point") -> "Vector":
        if not isinstance(other, Point):
            raise TypeError(
                "Can only substract a point from a point to generate a vector."
            )
        return Vector(*(self._pos - other._pos))

    def __eq__(self, other: "Point") -> bool:
        return np.array_equal(self._pos, other._pos)

    def __hash__(self):
        return hash((self._x, self._y))

    def __repr__(self) -> str:
        return "Point at ({x:.2f}, {y:.2f})".format(x=self._x, y=self._y)


class Vector(Point):
    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y)
        self._length = np.linalg.norm(self._pos)

    @property
    def length(self) -> float:
        return self._length

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(*(self._pos + other._pos))

    def __radd__(self, other) -> "Vector":
        if isinstance(other, int) and other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(*(self._pos - other._pos))

    def __neg__(self):
        return Vector(*(-self._pos))

    def __mul__(self, other: Union[float, "Vector"]) -> "Vector":
        if isinstance(other, Vector):
            return Vector(*(self._pos * other._pos))
        elif isinstance(other, (int, float)):
            return Vector(*(self._pos * other))
        else:
            raise TypeError(f"Unsupported type {type(other)} for __mul__")

    def __rmul__(self, other: Union[float, "Vector"]) -> "Vector":
        return self * other

    def __truediv__(self, a: float) -> "Vector":
        assert isinstance(a, (float, int))
        return Vector(*(self._pos / a))

    def dot(self, other: "Vector") -> float:
        return np.dot(self._pos, other._pos)

    def cross(self, other: "Vector") -> float:
        return np.cross(self._pos, other._pos)

    def __repr__(self) -> str:
        return "Vector ({x:.2f}, {y:.2f})".format(x=self._x, y=self._y)


class Line(Element):
    def __init__(self, a: Point, b: Point) -> None:
        self._a = a
        self._b = b
        self._envelop = (
            Point(*np.minimum(a._pos, b._pos)),
            Point(*np.maximum(a._pos, b._pos)),
        )

    def is_repel(self, other: "Line") -> bool:
        if all(
            (
                self._envelop[0]._x <= other._envelop[1]._x,
                self._envelop[1]._x >= other._envelop[0]._x,
                self._envelop[0]._y <= other._envelop[1]._y,
                self._envelop[1]._y >= other._envelop[0]._y,
            )
        ):
            return False
        return True

    def is_straddle(self, other: "Line") -> bool:
        """
        The criterion is "<=", which implies that the ends of one line can
        be "on" the extension cord of the other. Hence, two lines will be
        considered as "straddle" even if they overlap one another.
        """
        ac = other._a - self._a
        ad = other._b - self._a
        ab = self._b - self._a
        if ac.cross(ab) * ad.cross(ab) <= 0:
            return True
        return False

    def get_overlap(self, other: "Line") -> "Line":
        """
        self: ab   other: cd
        Preserves the direction for self.
        """

        def __check_zero_length(line: "Line"):
            if line._a == line._b:
                return None
            return line

        if self.is_repel(other):
            return None

        ac = other._a - self._a
        ad = other._b - self._a
        ab = self._b - self._a
        if ac.cross(ab) == 0 and ad.cross(ab) == 0:
            c_on = self.point_on(other._a)
            d_on = self.point_on(other._b)
            if c_on and d_on:
                return other
            if (not c_on) and (not d_on):
                return self

            on = other._a if c_on else other._b
            out = other._a if d_on else other._b
            ao = out - self._a
            if ab.dot(ao) > 0:
                return __check_zero_length(Line(on, self._b))
            return __check_zero_length(Line(self._a, on))
        return None

    def is_cross(self, other: "Line") -> bool:
        if self.is_repel(other):
            return False

        if all((self.is_straddle(other), other.is_straddle(self))):
            return True
        return False

    def cross_point(self, other: "Line") -> Union[bool, Point]:
        """
        Finds the cross point if it exists, returns if the two lines
        overlap is the cross point does not exist.

        self: ab   other: cd

        Args:
            other (Line): _description_

        Returns:
            Union[bool, Point]: (Point) if the cross point exists
                                (True) if they overlap
                                (False) if they do not cross over at all
        """
        ca = self._a - other._a
        cb = self._b - other._a
        da = self._a - other._b
        db = self._b - other._b
        S_abc = ca.cross(cb)  # 2 times the area size
        S_abd = da.cross(db)
        S_cda = ca.cross(da)
        S_cdb = cb.cross(db)

        if S_abc == S_abd == 0:
            ab = self._b - self._a
            cd = other._b - other._a
            # On the extension cord
            if self.is_repel(other):
                # Away from each other
                return False
            elif any(
                (
                    self._a == other._a and ab.dot(cd) < 0,
                    self._a == other._b and ab.dot(cd) > 0,
                )
            ):
                return self._a
            elif any(
                (
                    self._b == other._a and ab.dot(cd) > 0,
                    self._b == other._b and ab.dot(cd) < 0,
                )
            ):
                return self._b
            else:
                return True
        elif (S_abc * S_abd > 0) or (S_cda * S_cdb > 0):
            return False
        else:
            t = S_cda / (S_abd - S_abc)
            delta = t * (self._b - self._a)
            return self._a + delta

    def point_on(self, point: Point) -> bool:
        ac = point - self._a
        bc = point - self._b
        if all(
            (
                ac.cross(bc) == 0,
                ac.dot(bc) <= 0,
            )
        ):
            return True
        return False

    def __repr__(self) -> str:
        return "Line ({x1:.2f}, {y1:.2f}), ({x2:.2f}, {y2:.2f})".format(
            x1=self._a._x, y1=self._a._y, x2=self._b._x, y2=self._b._y
        )


class DirectEdge(Line):
    def __init__(self, a: Point, v: Vector) -> None:
        super().__init__(a, a + v)
        self._v = v

    def get_overlap(self, other: Line) -> "DirectEdge":
        """
        This method SHOULD ONLY BE CALLED when you know that they are overlapping!
        """
        raise NotImplementedError


class RelativePos(Enum):
    IN = -1
    ON = 0
    OUT = 1


class Polygon(Element):
    def __init__(self, vertices: List[Point]) -> None:
        assert len(vertices) >= 3
        self._vertices = vertices
        self._edges = [
            Line(a, b) for a, b in zip(vertices, vertices[1:] + [vertices[0]])
        ]
        for e, e_n in zip(self._edges, self._edges[1:] + [self._edges[0]]):
            if (e._b - e._a).cross(e_n._b - e_n._a) == 0:
                raise ValueError(f"Edge {e} and {e_n} are parallel!")
        self._envelop = (
            Point(*np.minimum.reduce([p._pos for p in vertices])),
            Point(*np.maximum.reduce([p._pos for p in vertices])),
        )

    def point_relative_pos(self, point: Point) -> Literal[-1, 0, 1]:
        """
        The logic is ported from (in Chinese):
        https://blog.csdn.net/zsjzliziyang/article/details/108813349
        """
        in_flag = False
        for edge in self._edges:
            a, b = edge._a, edge._b
            if edge.point_on(point):
                return RelativePos.ON

            if any(
                (
                    a._y < point._y and b._y >= point._y,
                    a._y >= point._y and b._y < point._y,
                )
            ):
                # The if above guarantees that a.y != b.y
                x_cross = a._x + (point._y - a._y) * ((b._x - a._x) / (b._y - a._y))
                if x_cross == point._x:
                    return RelativePos.ON
                if x_cross > point._x:
                    in_flag = not in_flag

        return RelativePos.IN if in_flag else RelativePos.OUT

    def __repr__(self) -> str:
        points = [str(p) for p in self._vertices]
        return "Polygon defined by (ordered) vertices: " + ", ".join(points)


class Box(Polygon):
    """
    0  ------------>  x

    |   a -- width -- b
    |
    |   |             |
    |   |           height
    v   |             |

    y   c ----------- d
    """

    def __init__(self, vertex: Point, width: float, height: float) -> None:
        ab = Vector(width, 0)
        ac = Vector(0, height)
        super().__init__([vertex, vertex + ab, vertex + ab + ac, vertex + ac])

    def point_relative_pos(self, point: Point) -> RelativePos:
        if all((
            (point._pos >= self._envelop[0]._pos).all(),
            (point._pos <= self._envelop[1]._pos).all(),
        )):
            for edge in self._edges:
                if edge.point_on(point):
                    return RelativePos.ON
            return RelativePos.IN
        return RelativePos.OUT

