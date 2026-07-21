#!/usr/bin/env python3
"""
Generate the richer FrugalNav arena: a bounded course with a pillar slalom, some
buildings, a dense marker field, a hard patch and a target. Emits BOTH the Gazebo
world SDF and a C++ header of the same coordinates, so the simulated obstacles and
the navigator's scene can never drift out of sync.

    python3 tools/gen_arena_world.py
writes:
    worlds/frugalnav_arena.world
    include/frugalnav/scene_arena.hpp
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

START = (58.0, 24.0)
TARGET = (0.0, 0.0)


def lerp(a, b, f):
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


def lateral(f, lat):
    seg = (TARGET[0] - START[0], TARGET[1] - START[1])
    n = math.hypot(*seg)
    axis = (seg[0] / n, seg[1] / n)
    left = (-axis[1], axis[0])
    p = lerp(START, TARGET, f)
    return (p[0] + lat * left[0], p[1] + lat * left[1])


# --- scene definition ---
PILLARS = [  # (x, y, radius) -- a slalom the auto-pilot must weave through
    (*lateral(0.30, 2.4), 1.5),
    (*lateral(0.44, -2.8), 1.5),
    (*lateral(0.56, 2.6), 1.8),   # the big one, near the hard patch
    (*lateral(0.70, -2.4), 1.5),
    (*lateral(0.82, 1.8), 1.3),
]
def _make_markers():
    # ArUco markers on a grid across the WHOLE field (not just the path), so an
    # absolute fix is available almost anywhere. Skip cells that hit a pillar.
    out = []
    for mx in range(4, 61, 7):
        for my in range(0, 31, 7):
            if all(math.hypot(mx - px, my - py) > pr + 2.2 for (px, py, pr) in PILLARS):
                out.append((float(mx), float(my)))
    return out


MARKERS = _make_markers()
HARD = (*lerp(START, TARGET, 0.38), 8.0)     # (x, y, radius)

# arena bounds (enclose the corridor with margin)
XMIN, XMAX, YMIN, YMAX = -8.0, 66.0, -10.0, 34.0
BUILDINGS = [  # (x, y, sx, sy, sz) scenery blocks off to the sides
    (30.0, 30.0, 6.0, 5.0, 4.0),
    (18.0, -6.0, 5.0, 6.0, 3.0),
    (46.0, -5.0, 7.0, 4.0, 5.0),
]


def wall(name, x, y, sx, sy):
    return f"""    <model name="{name}"><static>true</static><pose>{x} {y} 1.0 0 0 0</pose>
      <link name="l"><collision name="c"><geometry><box><size>{sx} {sy} 2.0</size></box></geometry></collision>
      <visual name="v"><geometry><box><size>{sx} {sy} 2.0</size></box></geometry>
      <material><ambient>0.20 0.24 0.30 1</ambient><diffuse>0.24 0.28 0.34 1</diffuse></material></visual></link></model>"""


def cylinder(name, x, y, r, h, rgba, z=None):
    z = h / 2 if z is None else z
    c = " ".join(map(str, rgba))
    return f"""    <model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} {z} 0 0 0</pose>
      <link name="l"><collision name="c"><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry></collision>
      <visual name="v"><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
      <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material></visual></link></model>"""


def box(name, x, y, sx, sy, sz, rgba):
    c = " ".join(map(str, rgba))
    return f"""    <model name="{name}"><static>true</static><pose>{x} {y} {sz/2} 0 0 0</pose>
      <link name="l"><collision name="c"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>
      <visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material></visual></link></model>"""


def flat_marker(name, x, y):
    return f"""    <model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} 0.03 0 0 0</pose>
      <link name="l"><visual name="v"><geometry><box><size>0.9 0.9 0.05</size></box></geometry>
      <material><ambient>0.95 0.95 0.98 1</ambient></material></visual></link></model>"""


def build_world():
    parts = []
    # perimeter walls
    t = 0.4
    parts.append(wall("wall_s", (XMIN+XMAX)/2, YMIN, XMAX-XMIN, t))
    parts.append(wall("wall_n", (XMIN+XMAX)/2, YMAX, XMAX-XMIN, t))
    parts.append(wall("wall_w", XMIN, (YMIN+YMAX)/2, t, YMAX-YMIN))
    parts.append(wall("wall_e", XMAX, (YMIN+YMAX)/2, t, YMAX-YMIN))
    # hard patch disc
    parts.append(f"""    <model name="hard_patch"><static>true</static><pose>{HARD[0]:.3f} {HARD[1]:.3f} 0.02 0 0 0</pose>
      <link name="l"><visual name="v"><geometry><cylinder><radius>{HARD[2]}</radius><length>0.02</length></cylinder></geometry>
      <material><ambient>0.96 0.62 0.04 0.22</ambient><diffuse>0.96 0.62 0.04 0.22</diffuse></material></visual></link></model>""")
    # pillars
    for i, (x, y, r) in enumerate(PILLARS):
        parts.append(cylinder(f"pillar{i}", x, y, r, 2.5, (0.33, 0.33, 0.36, 1)))
    # buildings
    for i, (x, y, sx, sy, sz) in enumerate(BUILDINGS):
        parts.append(box(f"building{i}", x, y, sx, sy, sz, (0.28, 0.30, 0.38, 1)))
    # markers
    for i, (x, y) in enumerate(MARKERS):
        parts.append(flat_marker(f"marker{i}", x, y))
    # target + start
    parts.append(cylinder("target_B", TARGET[0], TARGET[1], 0.6, 3.0,
                          (0.98, 0.75, 0.14, 1), z=1.5))
    parts.append(cylinder("start_pad", START[0], START[1], 1.0, 0.04,
                          (0.20, 0.83, 0.44, 0.85), z=0.02))

    body = "\n".join(parts)
    return f"""<?xml version="1.0" ?>
<!-- GENERATED by tools/gen_arena_world.py -- do not edit by hand. -->
<sdf version="1.6">
  <world name="frugalnav_arena">
    <gravity>0 0 -9.81</gravity>
    <scene>
      <ambient>0.45 0.47 0.5 1</ambient><background>0.62 0.66 0.72 1</background>
      <shadows>true</shadows>
      <fog><color>0.66 0.69 0.74 1</color><type>linear</type><start>18</start><end>85</end><density>0.02</density></fog>
    </scene>
    <light name="sun" type="directional"><cast_shadows>true</cast_shadows>
      <pose>0 0 40 0 0 0</pose><diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular><direction>-0.4 0.3 -0.9</direction></light>
    <model name="ground"><static>true</static><link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>300 300</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>300 300</size></plane></geometry>
      <material><ambient>0.10 0.12 0.16 1</ambient><diffuse>0.14 0.17 0.22 1</diffuse></material></visual></link></model>
{body}
    <!-- services for teleport / reset -->
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros><update_rate>0.0</update_rate>
    </plugin>
    <physics name="ode" type="ode"><max_step_size>0.004</max_step_size>
      <real_time_update_rate>250</real_time_update_rate></physics>
  </world>
</sdf>
"""


def build_header():
    def vecP(lst):
        return ", ".join(f"{{{x:.3f}f,{y:.3f}f,{r:.3f}f}}" for (x, y, r) in lst)
    def vecM(lst):
        return ", ".join(f"{{{x:.3f}f,{y:.3f}f}}" for (x, y) in lst)
    return f"""// GENERATED by tools/gen_arena_world.py -- do not edit by hand.
// The arena scene shared by the Gazebo world and the navigator (kept in sync).
#ifndef FRUGALNAV_SCENE_ARENA_HPP
#define FRUGALNAV_SCENE_ARENA_HPP
#include <array>
#include <vector>
namespace frugalnav {{ namespace scene {{
struct Pillar {{ float x, y, r; }};
constexpr float START_X = {START[0]:.3f}f, START_Y = {START[1]:.3f}f;
constexpr float TARGET_X = {TARGET[0]:.3f}f, TARGET_Y = {TARGET[1]:.3f}f;
constexpr float HARD_X = {HARD[0]:.3f}f, HARD_Y = {HARD[1]:.3f}f, HARD_R = {HARD[2]:.3f}f;
inline const std::vector<Pillar> PILLARS = {{ {vecP(PILLARS)} }};
inline const std::vector<std::array<float,2>> MARKERS = {{ {vecM(MARKERS)} }};
}} }}
#endif
"""


if __name__ == "__main__":
    with open(os.path.join(PKG, "worlds", "frugalnav_arena.world"), "w") as f:
        f.write(build_world())
    os.makedirs(os.path.join(PKG, "include", "frugalnav"), exist_ok=True)
    with open(os.path.join(PKG, "include", "frugalnav", "scene_arena.hpp"), "w") as f:
        f.write(build_header())
    print("wrote worlds/frugalnav_arena.world and include/frugalnav/scene_arena.hpp")
    print(f"  {len(PILLARS)} pillars, {len(MARKERS)} markers")
