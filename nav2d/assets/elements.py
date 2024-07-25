from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Literal, Any
from enum import Enum
import numpy as np


DEFAULT_DTYPE = np.float64


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
        self._pos = np.array([x, y], dtype=DEFAULT_DTYPE)
        
    @property
    def x(self) -> float:
        return self._x
    
    @x.setter
    def x(self, x: float) -> None:
        self._x = x
        self._pos[0] = x
    
    @property
    def y(self) -> float:
        return self._y
    
    @y.setter
    def y(self, y: float) -> None:
        self._y = y
        self._pos[1] = y

    @property
    def pos(self) -> np.array:
        return self._pos
    
    @pos.setter
    def pos(self, pos: np.array) -> None:
        self._pos = pos
        self._x, self._y = pos

    # Handle this case in Vector.__radd__ because Python MRO treats Vector as a
    # subclass of Point and resolve Point + Vector with Vector.__radd__.
    # def __add__(self, vector: "Vector") -> "Point":
    #     if not isinstance(vector, Vector):
    #         raise TypeError("Can only add a vector to a point.")
    #     return Point(*(self._pos + vector._pos))

    def __sub__(self, other: "Point") -> "Vector":
        if not isinstance(other, Point):
            raise TypeError(
                "Can only substract a point from a point to generate a vector."
            )
        return Vector(*(self._pos - other._pos))

    def __eq__(self, other: "Point") -> bool:
        # return np.array_equal(self._pos, other._pos)
        return np.allclose(self._pos, other._pos, atol=1e-6)

    def __hash__(self):
        return hash((self._x, self._y))

    def __repr__(self) -> str:
        return "Point({x:.2f}, {y:.2f})".format(x=self._x, y=self._y)


class Vector(Point):
    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y)
        self._length = np.linalg.norm(self._pos)

    @property
    def length(self) -> float:
        return self._length

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(*(self._pos + other._pos))

    def __radd__(self, other) -> Union["Vector", "Point"]:
        if isinstance(other, int) and other == 0:
            return self
        if isinstance(other, Point):
            return Point(*(other._pos + self._pos))
        return self.__add__(other)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(*(self._pos - other._pos))

    def __neg__(self):
        return Vector(*(-self._pos))

    def __mul__(self, other: Union[float, "Vector"]) -> "Vector":
        if isinstance(other, Vector):
            return Vector(*(self._pos * other._pos))
        elif isinstance(other, (int, float, DEFAULT_DTYPE)):
            return Vector(*(self._pos * other))
        else:
            raise TypeError(f"Unsupported type {type(other)} for __mul__")

    def __rmul__(self, other: Union[float, "Vector"]) -> "Vector":
        return self * other

    def __truediv__(self, a: float) -> "Vector":
        assert isinstance(a, (float, int, DEFAULT_DTYPE))
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
        self._envelope = (
            Point(*np.minimum(a.pos, b.pos)),
            Point(*np.maximum(a.pos, b.pos)),
        )

    @property
    def a(self) -> Point:
        return self._a
    
    @property
    def b(self) -> Point:
        return self._b
    
    @property
    def envelope(self) -> Tuple[Point, Point]:
        return self._envelope

    def is_repel(self, other: "Line") -> bool:
        if all(
            (
                self._envelope[0].x <= other._envelope[1].x,
                self._envelope[1].x >= other._envelope[0].x,
                self._envelope[0].y <= other._envelope[1].y,
                self._envelope[1].y >= other._envelope[0].y,
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
            x1=self._a.x, y1=self._a.y, x2=self._b.x, y2=self._b.y
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
    
    
class Shape(Element):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self._envelope = self._get_envelope()
        self._np_random = np.random.default_rng(seed=seed)
        
    @abstractmethod
    def _get_envelope(self) -> Tuple[Point, Point]: ...
    
    @property
    def envelope(self) -> Tuple[Point, Point]:
        return self._envelope

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            print('Reinitialize np_random')
            self._np_random = np.random.default_rng()
        return self._np_random

    @np_random.setter
    def np_random(self, np_random: np.random.Generator) -> None:
        self._np_random = np_random

    @abstractmethod
    def __add__(self, vector: Vector) -> "Shape": ...
    
    def __radd__(self, x: Any) -> None:
        raise TypeError("Can only use __add__ to add a vector to a shape.")
    
    @abstractmethod
    def move(self, vector: Vector) -> None: ...
    
    @abstractmethod
    def scale(self) -> None: ...
    
    @abstractmethod
    def point_relative_pos(self, point: Point) -> RelativePos: ...
    
    @abstractmethod
    def sample_point(self) -> Point:
        """ Sample a point INSIDE the shape. """
        ...
    
    
class Circle(Shape):
    def __init__(self, center: Point, radius: float, seed: int = 0) -> None:
        self._c = center
        self._r = radius
        super().__init__(seed=seed)
        
    def _get_envelope(self) -> Tuple[Point]:
        return (
            Point(self._c.x - self._r, self._c.y - self._r),
            Point(self._c.x + self._r, self._c.y + self._r),
        )

    @property
    def center(self) -> Point:
        return self._c
    
    @property
    def radius(self) -> float:
        return self._r
        
    def __add__(self, vector: Vector) -> "Circle":
        return Circle(self._c + vector, self._r)
    
    def move(self, vector: Vector) -> None:
        self._c += vector
    
    def scale(self, ratio: float) -> None:
        self._r *= ratio
    
    def point_relative_pos(self, point: Point) -> RelativePos:
        dist = (point - self._c).length
        if dist < self._r:
            return RelativePos.IN
        if dist == self._r:
            return RelativePos.ON
        return RelativePos.OUT
    
    def sample_point(self) -> Point:
        r = self._r * (self.np_random.uniform() ** 2)
        theta = self.np_random.uniform(0, 2 * np.pi)
        return Point(self._c.x + r * np.cos(theta), self._c.y + r * np.sin(theta))
    
    def __repr__(self) -> str:
        return f"Circle with center {self._c} and radius {self._r}"


class Polygon(Shape):
    def __init__(self, vertices: List[Point], seed: int = 0) -> None:
        assert len(vertices) >= 3
        self._vertices = vertices
        self._edges = [
            Line(a, b) for a, b in zip(vertices, vertices[1:] + [vertices[0]])
        ]
        for e, e_n in zip(self._edges, self._edges[1:] + [self._edges[0]]):
            if (e.b - e.a).cross(e_n.b - e_n.a) == 0:
                raise ValueError(f"Edge {e} and {e_n} are parallel!")
        super().__init__(seed=seed)
        
    def _get_envelope(self) -> Tuple[Point]:
        return (
            Point(*np.minimum.reduce([p.pos for p in self._vertices])),
            Point(*np.maximum.reduce([p.pos for p in self._vertices])),
        )

    @property
    def vertices(self) -> List[Point]:
        return self._vertices
    
    @property
    def edges(self) -> List[Line]:
        return self._edges
    
    def __add__(self, vector: Vector) -> "Polygon":
        return Polygon([p + vector for p in self._vertices])
    
    def move(self, vector: Vector) -> None:
        for p in self._vertices:
            p += vector

    def scale(self, new_envelope: Tuple[Point, Point]) -> None:
        (min_x, min_y), (max_x, max_y) = new_envelope[0].pos, new_envelope[1].pos
        (min_x_p, min_y_p), (max_x_p, max_y_p) = self._envelope[0].pos, self._envelope[1].pos
        
        scale_x = (max_x - min_x) / (max_x_p - min_x_p)
        scale_y = (max_y - min_y) / (max_y_p - min_y_p)
        
        for p in self._vertices:
            p.x = min_x + (p.x - min_x_p) * scale_x
            p.y = min_y + (p.y - min_y_p) * scale_y
        self._envelope = new_envelope

    def point_relative_pos(self, point: Point) -> Literal[-1, 0, 1]:
        """
        The logic is ported from (in Chinese):
        https://blog.csdn.net/zsjzliziyang/article/details/108813349
        """
        in_flag = False
        for edge in self._edges:
            a, b = edge.a, edge.b
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
    
    def sample_point(self) -> Point:
        min_x, min_y = self._envelope[0].pos
        max_x, max_y = self._envelope[1].pos
        while True:
            x = self.np_random.uniform(min_x, max_x)
            y = self.np_random.uniform(min_y, max_y)
            point = Point(x, y)
            if self.point_relative_pos(point) == RelativePos.IN:
                return point

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

    def __init__(
        self, vertex: Point, width: float, height: float, seed: int = 0,
    ) -> None:
        ab = Vector(width, 0)
        ac = Vector(0, height)
        super().__init__(
            [vertex, vertex + ab, vertex + ab + ac, vertex + ac],
            seed=seed,
        )

    def point_relative_pos(self, point: Point) -> RelativePos:
        if all((
            (point.pos >= self._envelope[0].pos).all(),
            (point.pos <= self._envelope[1].pos).all(),
        )):
            for edge in self._edges:
                if edge.point_on(point):
                    return RelativePos.ON
            return RelativePos.IN
        return RelativePos.OUT
    
    def sample_point(self) -> Point:
        min_x, min_y = self._envelope[0].pos
        max_x, max_y = self._envelope[1].pos
        return Point(self.np_random.uniform(min_x, max_x), self.np_random.uniform(min_y, max_y))

