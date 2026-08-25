#!/usr/bin/env python3
"""Beam twin of probes/ground-truth.pov — same camera (mirrored), same objects.
Text transforms derived by conjugating POV net chains through the p2b mirror:
  red   "POV-Ray"     net Rx(90)Rz(-90): beam rotY(+90), west wall (x=1.9)
  green "Version 3.8" net Rx(90) only:   beam NO rotation, south wall (z~1.5)
Anchors tuned to match POV ground-truth centroids:
  red cx=0.128 cy=0.601 | green cx=0.765 cy=0.613
  yellow origin 0.492/0.973 | blue +z 0.495/0.549  (camera equivalence check)
"""
import json, math

def p2b(v):
    return [-v[0], v[2], -v[1]]

scene = {
    "version": 1,
    "camera": {"pos": p2b((3.2, 3.2, 1.8)), "lookAt": p2b((-1, -1, 0.9)), "fov": 45},
    "environment": {"color": [0.29, 0.28, 0.30], "ambient": 0.9, "ambientColor": [1, 1, 1]},
    "render": {"width": 640, "height": 360, "samples": 1, "output": "gt-beam.png"},
    "materials": {
        "wall":  {"type": "matte", "color": [0.55, 0.52, 0.50]},
        "floor": {"type": "matte", "color": [0.50, 0.45, 0.40]},
        "red":   {"type": "matte", "color": [1.0, 0.10, 0.10]},
        "green": {"type": "matte", "color": [0.10, 1.0, 0.10]},
        "yellow": {"type": "matte", "color": [1, 1, 0.05]},
        "cyan":  {"type": "matte", "color": [0.05, 1, 1]},
        "magenta": {"type": "matte", "color": [1, 0.05, 1]},
        "blue":  {"type": "matte", "color": [0.2, 0.4, 1.0]},
    },
    "lights": [
        {"type": "directional", "dir": p2b((0.7, -0.83, -0.24)),
         "color": [1, 1, 1], "intensity": 1.2, "shadowless": True},
    ],
    "objects": [
        {"type": "plane", "point": [0, -0.3, 0], "normal": [0, 1, 0], "material": "floor"},
        # west wall (pov x -2.0..-1.9) and south wall (pov y -1.7..-1.6)
        {"type": "box", "min": [-3.2, -6.0, -1.7], "max": [-1.9, 0.5, 1.7], "material": "wall"},   # west (pov x=-2)
        {"type": "box", "min": [-2.0, -6.0, -1.7], "max": [2.0, 0.5, -1.6], "material": "wall"},   # south (pov y=-1.6)
        # red text: rotY(90), face flush west wall x=1.9, baseline y=0.575
        {"type": "text", "text": "POV-Ray", "size": 0.21, "depth": 0.075, "material": "red",
         "transform": {"rotate": [0, 90, 0], "translate": [1.9, 0.575, -0.55]}},
        # green text: NO rotation, south wall, face toward camera (-z)
        {"type": "text", "text": "Version 3.8", "size": 0.21, "depth": 0.075, "material": "green",
         "transform": {"translate": [-0.15, 0.575, 1.42]}},
        # axis markers
        {"type": "sphere", "center": [0, 0, 0], "radius": 0.08, "material": "yellow"},
        {"type": "sphere", "center": [-1, 0, 0], "radius": 0.08, "material": "cyan"},
        {"type": "sphere", "center": [0, 0, -1], "radius": 0.08, "material": "magenta"},
        {"type": "sphere", "center": [0, 1, 0], "radius": 0.08, "material": "blue"},
    ],
}
json.dump(scene, open("gt-beam.json", "w"), indent=1)
print("gt-beam.json written,", len(scene["objects"]), "objects")
