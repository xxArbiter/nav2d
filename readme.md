# 2D Navigation Gym Environment

This repository implements a [Gym](https://github.com/openai/gym) environment for simulating simple 2D navigation scenarios.

### Installation
Activate a virtual environment or a conda environment. Then, run the following command:
```
pip install nav2d
```

### Use cases
The building blocks in this toolbox are `Point`, `Vector`, and `Polygon` defined in `nav2d.assets.elements`.

- Point: `p = Point(x, y)`, `p.pos` is a `np.array`. You can add a `Vector` to a `Point` to make it a new `Point`.
- Vector: inherits from `Point`. You can apply various algebraic operations to `Vector`. Many operators are overloaded there, e.g., `sum(List[Vector])` gives you a singe `Vector`. `Vector` helps you manipulate multiple movements.
- Polygon: `zone = Polygon(List[Point])`. Given a zone and a point, `zone.point_relative_pos(p)` tells you their relative position `IN | ON | OUT`. A `Polygon` is defined by its vertices (`Point`s).

### Authors

[Xiaoyu Wang](https://www.linkedin.com/in/xiaoyu-wang-8b938b203/?originalSubdomain=ca)

[Jihwan Jeong](https://jihwan-jeong.netlify.app/)
