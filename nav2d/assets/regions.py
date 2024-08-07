from abc import abstractmethod
from typing import Tuple, List, Dict, Callable, Union
from collections import defaultdict
import numpy as np
import pygame

from .elements import Element, Point, Vector, Line, Polygon, RelativePos


class Region(Element):
    def __init__(self, zone: Polygon) -> None:
        self._zone = zone
        
    @property
    def zone(self):
        return self._zone

    def check_segment_cross(
        self, p0: Point, proposition: Vector
    ) -> Tuple[List[Point], List[List[Line]], List[RelativePos]]:
        # def __handle_corners(edge: Line, edge_next: Line):
        #     """
        #     Determine if we should consider the corner as an anchor point.
            
        #     Return False if the proposition vector cross the corner from the
        #     outside of the polygon.
        #     """
        #     if edge.b == p0:
        #         return True
        #     ba = edge.a - edge.b
        #     bc = edge_next.b - edge_next.a
        #     if proposition.cross(ba) * proposition.cross(bc) > 0:
        #         return False
        #     return True
        
        anchor_edges: Dict[Point, List[Line]] = defaultdict(list)
        pp = Line(p0, p0 + proposition)
        for edge in self._zone._edges:
            cross = pp.cross_point(edge)
            if cross is True:
                overlap = pp.get_overlap(edge)
                anchor_edges[overlap.a].append(edge)
                anchor_edges[overlap.b].append(edge)
            elif isinstance(cross, Point):
                anchor_edges[cross].append(edge)
                
        anchor_pts = list(anchor_edges.keys()) + [p0, p0 + proposition]
        anchor_pts = list(set(anchor_pts))
        lengths = [(cross - p0).length for cross in anchor_pts]
        _, anchor_pts = zip(*sorted(zip(lengths, anchor_pts), key=lambda x: x[0]))
        
        cross_edges = [list(dict.fromkeys(anchor_edges[anchor])) for anchor in anchor_pts]
        assert all([len(edges) <= 2 for edges in cross_edges]), \
            f"An anchor point can only cross at most 2 edges. {cross_edges}, \n {anchor_edges}"
        assert all([len(edges) > 0 for edges in cross_edges[1:-1]]), (
            "An anchor point other than the two ends should cross at least one edge.\n" +
            f"p0: {p0}, proposition: {proposition}\n" +
            f"polygon: {self._zone},\n" +
            f"anchor_pts: {anchor_pts},\n" +
            f"cross_edges: {cross_edges}"
        )
        
        seg2poly_relation = []
        for p1, p2 in zip(anchor_pts[:-1], anchor_pts[1:]):
            mid = p1 + (p2 - p1) / 2
            seg2poly_relation.append(self._zone.point_relative_pos(mid))

        return anchor_pts, cross_edges, seg2poly_relation

        # anchor_pts: List[Point] = [p0, p0 + proposition]
        # pp = Line(p0, p0 + proposition)
        # for edge, edge_next in zip(
        #     self._zone._edges, self._zone._edges[1:] + [self._zone._edges[0]]
        # ):
        #     cross = pp.cross_point(edge)
        #     if cross is True:
        #         overlap = pp.get_overlap(edge)
        #         anchor_pts.extend([overlap.a, overlap.b])

        #     elif isinstance(cross, Point):
        #         # Vertex crosspoint will only be handled on the corner.
        #         if cross == edge.b:
        #             if any((
        #                 cross == p0,
        #                 __handle_corners(edge, edge_next),
        #             )):
        #                 anchor_pts.append(edge.b)
        #         elif cross == p0:
        #             continue
        #         elif cross != edge.a:
        #             anchor_pts.append(cross)
                

        # anchor_pts = list(set(anchor_pts))
        # lengths = [(cross - p0).length for cross in anchor_pts]
        # _, anchor_pts = zip(*sorted(zip(lengths, anchor_pts), key=lambda x: x[0]))
        # seg2poly_relation = []
        # for p1, p2 in zip(anchor_pts[:-1], anchor_pts[1:]):
        #     mid = p1 + (p2 - p1) / 2
        #     seg2poly_relation.append(self._zone.point_relative_pos(mid))

        # return anchor_pts, seg2poly_relation

    @property
    @abstractmethod
    def should_apply(self): ...

    def first_contact(
        self, p0: Point, propositions: List[Vector]
    ) -> Tuple[Point, float]:
        if isinstance(propositions, Vector):
            propositions = [propositions]

        if self._zone.point_relative_pos(p0) == RelativePos.IN:
            return p0, -1

        p = p0
        cuml_dist = 0
        for prop in propositions:
            anchor_pts, _, rel_poss = self.check_segment_cross(p, prop)
            for anchor1, anchor2, rel_pos in zip(
                anchor_pts[:-1], anchor_pts[1:], rel_poss
            ):
                if rel_pos in self.should_apply:
                    return anchor1, cuml_dist
                cuml_dist += (anchor2 - anchor1).length
            p += prop

        return None, np.inf

    def render(self, viewer, to_pixel: Callable, color: Tuple[int, int, int] = None):
        pygame.draw.polygon(
            viewer,
            self.COLOR if color is None else color,
            [to_pixel(v.pos) for v in self._zone._vertices],
            width=0,  # 0 for fill, > 0 for line thickness
        )


class DynamicRegion(Region):
    COLOR = [160, 160, 160]

    @abstractmethod
    def apply_dynamic(self, p0: Point, propositions: List[Vector]): ...

    @property
    def should_apply(self):
        return [RelativePos.IN]


class SimpleDynamicRegion(DynamicRegion):
    def apply_dynamic(self, p0: Point, proposition: Vector):
        if not isinstance(proposition, Vector):
            raise TypeError("SimpleDynamicRegion does not support complex movements.")
        return self._apply_dynamic(p0, proposition)

    @abstractmethod
    def _apply_dynamic(self, p0: Point, proposition: Vector): ...


class PunchRegion(SimpleDynamicRegion):
    """
    An extra, constant motion will be applied as long as the agent ENTERS the region.
    """

    def __init__(self, zone: Polygon, force: Vector) -> None:
        super().__init__(zone)
        self._force = force

    def _apply_dynamic(self, p0: Point, proposition: Vector):
        _, _, rel_poss = self.check_segment_cross(p0, proposition)
        if RelativePos.IN in rel_poss:
            return proposition + self._force
        return proposition

    def __repr__(self) -> str:
        return f"Punch zone with a fixed force '{self._force.pos}'. " + str(self._zone)


class NoEntryRegion(DynamicRegion):
    """ The agent CAN walk alongside the walls. """

    COLOR = [0, 0, 0]
    BOUNCE_BACK_DIST = 1e-3
    BOUNCE_REFLECTION_FACTOR = 0.5
    
    def __init__(self, zone: Polygon, bounce: Union[None, str] = None) -> None:
        if bounce is None:
            self._bounce = self._bounce_reject
        elif bounce == "reflection":
            self._bounce = self._bounce_reflection
        elif bounce == "back":
            self._bounce = self._bounce_back
        else:
            raise ValueError(f"Unknown bounce type: {bounce}")
        super().__init__(zone)
    
    def _bounce_reflection(
        self,
        movements: List[Vector],
        intention: Vector,
        edges: List[Line],
        bounce_dist: float
    ):
        assert len(edges) > 0, "No edge to bounce (reflection)."
        bounce_dist *= NoEntryRegion.BOUNCE_REFLECTION_FACTOR
        
        if len(edges) == 1:
            edge_vec = edges[0].a - edges[0].b
            normal = Vector(edge_vec.y, -edge_vec.x)
        if len(edges) == 2:
            # We need to find the "normal" (angle bisector) vector of the corner.
            assert edges[0].a == edges[1].b or edges[0].b == edges[1].a
            normal = (edges[0].a - edges[0].b) + (edges[1].b - edges[1].a)

        normal /= normal.length
        reflection = intention - 2 * intention.dot(normal) * normal
        reflection = reflection / reflection.length * bounce_dist
        movements.append(reflection)
        
        return movements
    
    def _bounce_back(self, movements: List[Vector], **kwargs):
        to_bounce = NoEntryRegion.BOUNCE_BACK_DIST
        
        while len(movements) > 0:
            if movements[-1].length < to_bounce:
                to_bounce -= movements[-1].length
                movements.pop()
            else:
                movements[-1] -= movements[-1] / movements[-1].length * to_bounce
                return movements
            
        return movements
    
    def _bounce_reject(self, movements: List[Vector], **kwargs):
        return movements

    def apply_dynamic(self, p0: Point, propositions: List[Vector]):
        p = p0
        movements = []
        total_length = sum([prop.length for prop in propositions])
        cmlt_length = 0
        for prop in propositions:
            anchor_pts, cross_edges, rel_poss = self.check_segment_cross(p, prop)
            for anchor, edges, rel_pos in zip(anchor_pts[:-1], cross_edges[:-1], rel_poss):
                if rel_pos in self.should_apply:
                    movements.append(anchor - p)
                    cmlt_length += (anchor - p).length
                    return self._bounce(
                        movements=movements,
                        intention=prop,
                        edges=edges,
                        bounce_dist=total_length - cmlt_length,
                    )
            movements.append(prop)
            cmlt_length += prop.length
            p += prop
        return movements

    def __repr__(self) -> str:
        return "No entry zone. " + str(self._zone)
    
    
class Wall(DynamicRegion):
    """ The agent CAN walk alongside the walls. """

    COLOR = [0, 0, 0]
    
    def __init__(self, zone: Line) -> None:
        super().__init__(zone)
        
    def apply_dynamic(self, p0: Point, propositions: List[Vector]):
        p = p0
        movements = []
        for prop in propositions:
            line = Line(p, p + prop)
            cross = line.cross_point(self._zone)
            if isinstance(cross, Point):
                movements.append(cross - p)
                return movements
            movements.append(prop)
            p += prop
        return movements
    
    def __repr__(self) -> str:
        return "Wall. " + str(self._zone)
            


class SlipperyRegion(DynamicRegion):
    
    COLOR = [0xFF, 0xFF, 0x9F]
    
    def __init__(self, zone: Polygon, force: Vector, **kwargs) -> None:
        super().__init__(zone)
        self._force = force

    def apply_dynamic(self, p0: Point, propositions: List[Vector]):
        p = p0
        movements = []
        total_length = sum([prop.length for prop in propositions])
        for prop in propositions:
            normal_length = 0
            anchor_pts, _, rel_poss = self.check_segment_cross(p, prop)
            movement = None
            for anchor1, anchor2, rel_pos in zip(
                anchor_pts[:-1], anchor_pts[1:], rel_poss
            ):
                if rel_pos not in self.should_apply:
                    normal_length += (anchor2 - anchor1).length
                else:
                    delta_t = (prop.length - normal_length) / total_length
                    movement = [] if anchor1 == p else [anchor1 - p]
                    movement.append(p + prop - anchor1 + self._force * delta_t)
                    break
            movement = [prop] if movement is None else movement
            movements.extend(movement)
            for m in movement:
                p += m

        return movements

    def __repr__(self) -> str:
        return f"Slippery zone with a force rate '{self._force.pos}'. " + str(
            self._zone
        )


class BlackHoleRegion(DynamicRegion):
    def __init__(self, zone: Polygon, center: Point, force: float) -> None:
        super().__init__(zone)
        self._center = center
        self._force = force

    def apply_dynamic(self, p0: Point, propositions: List[Vector]):
        p = p0
        movements = []
        total_length = sum([prop.length for prop in propositions])
        for prop in propositions:
            normal_length = 0
            anchor_pts, _, rel_poss = self.check_segment_cross(p, prop)
            movement = None
            for anchor1, anchor2, rel_pos in zip(
                anchor_pts[:-1], anchor_pts[1:], rel_poss
            ):
                if rel_pos not in self.should_apply:
                    normal_length += (anchor2 - anchor1).length
                else:
                    vector = self._center - anchor1
                    direction = vector / vector.length
                    delta_t = (prop.length - normal_length) / total_length
                    movement = [] if anchor1 == p else [anchor1 - p]
                    applied_force = (
                        vector
                        if vector.length
                        < self._force
                        * delta_t  # The final movement should not exceed the center.
                        else direction * self._force * delta_t
                    )
                    movement.append(p + prop - anchor1 + applied_force)
                    break
            movement = [prop] if movement is None else movement
            movements.extend(movement)
            for m in movement:
                p += m

        return movements

    def __repr__(self) -> str:
        return (
            f"Black hold zone centered on {self._center} with a force rate {self._force}. "
            + str(self._zone)
        )


class RewardRegion(Region):
    """Reward WILL BE applied when the agent touches the boundary."""

    COLOR = [255, 102, 102]

    def __init__(self, zone: Polygon, reward: float) -> None:
        super().__init__(zone)
        self._reward = reward

    def apply_reward(self, p0: Point, propositions: List[Vector]):
        p = p0
        total_length = 0
        counted_length = 0
        for prop in propositions:
            total_length += prop.length
            anchor_pts, _, rel_poss = self.check_segment_cross(p, prop)
            for anchor1, anchor2, rel_pos in zip(
                anchor_pts[:-1], anchor_pts[1:], rel_poss
            ):
                if rel_pos in self.should_apply:
                    counted_length += (anchor2 - anchor1).length
            p += prop

        if total_length == 0:
            rel_pos = self._zone.point_relative_pos(p0)
            if rel_pos in self.should_apply:
                return self._reward
            return 0

        return self._reward * counted_length / total_length

    @property
    def should_apply(self):
        return [RelativePos.IN, RelativePos.ON]

    def __repr__(self) -> str:
        return f"Reward zone with reward rate '{self._reward}'. " + str(self._zone)


class SimpleRewardRegion(RewardRegion):
    """Penalizes the agent as long as it touches the region."""

    def apply_reward(self, p0: Point, propositions: List[Vector]):
        rel_pos = self._zone.point_relative_pos(p0 + sum(propositions))
        if rel_pos in self.should_apply:
            return self._reward
        return 0

    def __repr__(self) -> str:
        return f"Simple reward zone with constant reward '{self._reward}/step'. " + str(
            self._zone
        )

