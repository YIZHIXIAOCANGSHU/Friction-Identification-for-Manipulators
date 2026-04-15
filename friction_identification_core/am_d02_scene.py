from __future__ import annotations

"""Build an augmented MuJoCo scene from the AM-D02 URDF description."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


_PI_HALF = "1.5707963"


def _ensure_child(root: ET.Element, tag: str) -> ET.Element:
    """Return an existing XML child or create it if missing."""

    node = root.find(tag)
    if node is None:
        node = ET.SubElement(root, tag)
    return node


def _add_axis_geom(
    parent: ET.Element,
    *,
    axis: str,
    radius: float,
    half_length: float,
    color: str,
) -> None:
    """Attach a colored axis cylinder to an MJCF body."""

    geom = ET.SubElement(parent, "geom")
    geom.set("type", "cylinder")
    geom.set("size", f"{radius} {half_length}")
    geom.set("rgba", color)
    geom.set("contype", "0")
    geom.set("conaffinity", "0")
    geom.set("mass", "0")

    if axis == "x":
        geom.set("pos", f"{half_length} 0 0")
        geom.set("euler", f"0 {_PI_HALF} 0")
    elif axis == "y":
        geom.set("pos", f"0 {half_length} 0")
        geom.set("euler", f"-{_PI_HALF} 0 0")
    else:
        geom.set("pos", f"0 0 {half_length}")


def _add_mocap_marker(
    worldbody: ET.Element,
    *,
    body_name: str,
    box_size: float,
    box_rgba: str,
    axis_radius: float,
    axis_half_length: float,
) -> None:
    """Add a mocap marker body used to visualize target poses in MuJoCo."""

    body = ET.SubElement(worldbody, "body")
    body.set("name", body_name)
    body.set("pos", "0 0 0")
    body.set("mocap", "true")

    geom = ET.SubElement(body, "geom")
    geom.set("type", "box")
    geom.set("size", f"{box_size} {box_size} {box_size}")
    geom.set("rgba", box_rgba)
    geom.set("contype", "0")
    geom.set("conaffinity", "0")
    geom.set("mass", "0")

    _add_axis_geom(body, axis="x", radius=axis_radius, half_length=axis_half_length, color="1 0 0 0.8")
    _add_axis_geom(body, axis="y", radius=axis_radius, half_length=axis_half_length, color="0 1 0 0.8")
    _add_axis_geom(body, axis="z", radius=axis_radius, half_length=axis_half_length, color="0 0 1 0.8")


def _attach_tcp_body(worldbody: ET.Element, tcp_offset: np.ndarray) -> None:
    """Attach a visible TCP helper body to the URDF's terminal link."""

    for body in worldbody.iter("body"):
        if body.get("name") != "ArmLseventh_Link":
            continue

        tcp_body = ET.SubElement(body, "body")
        tcp_body.set("name", "tcp")
        tcp_body.set("pos", f"{tcp_offset[0]} {tcp_offset[1]} {tcp_offset[2]}")
        _add_axis_geom(tcp_body, axis="x", radius=0.003, half_length=0.05, color="1 0 0 1")
        _add_axis_geom(tcp_body, axis="y", radius=0.003, half_length=0.05, color="0 1 0 1")
        _add_axis_geom(tcp_body, axis="z", radius=0.003, half_length=0.05, color="0 0 1 1")
        return

    raise ValueError("Failed to find ArmLseventh_Link while attaching the TCP body.")


def _normalize_mesh_paths(root: ET.Element) -> None:
    """Convert package-style URDF mesh paths to filenames MuJoCo can resolve."""

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and filename.startswith("package://"):
            mesh.set("filename", os.path.basename(filename))


def _augment_scene(root: ET.Element, tcp_offset: np.ndarray) -> None:
    """Inject lights, floor, markers, and TCP helpers into the exported MJCF."""

    asset = _ensure_child(root, "asset")

    grid_texture = ET.SubElement(asset, "texture")
    grid_texture.set("name", "grid_tex")
    grid_texture.set("type", "2d")
    grid_texture.set("builtin", "checker")
    grid_texture.set("rgb1", "0.4 0.4 0.4")
    grid_texture.set("rgb2", "0.3 0.3 0.3")
    grid_texture.set("width", "512")
    grid_texture.set("height", "512")

    grid_material = ET.SubElement(asset, "material")
    grid_material.set("name", "grid_mat")
    grid_material.set("texture", "grid_tex")
    grid_material.set("texrepeat", "8 8")
    grid_material.set("reflectance", "0.05")

    visual = _ensure_child(root, "visual")
    headlight = _ensure_child(visual, "headlight")
    headlight.set("ambient", "0.15 0.15 0.15")
    headlight.set("diffuse", "0.35 0.35 0.35")
    headlight.set("specular", "0.1 0.1 0.1")

    worldbody = _ensure_child(root, "worldbody")

    floor = ET.SubElement(worldbody, "geom")
    floor.set("name", "floor")
    floor.set("type", "plane")
    floor.set("size", "3 3 0.1")
    floor.set("material", "grid_mat")
    floor.set("contype", "0")
    floor.set("conaffinity", "0")

    key_light = ET.SubElement(worldbody, "light")
    key_light.set("name", "key_light")
    key_light.set("pos", "0 -1.5 3")
    key_light.set("dir", "0 0.4 -1")
    key_light.set("diffuse", "0.4 0.4 0.4")
    key_light.set("directional", "true")

    fill_light = ET.SubElement(worldbody, "light")
    fill_light.set("name", "fill_light")
    fill_light.set("pos", "2 2 2.5")
    fill_light.set("dir", "-0.5 -0.5 -1")
    fill_light.set("diffuse", "0.2 0.2 0.2")
    fill_light.set("directional", "true")

    _add_mocap_marker(
        worldbody,
        body_name="target_pose",
        box_size=0.014,
        box_rgba="0.2 0.8 0.2 0.35",
        axis_radius=0.002,
        axis_half_length=0.05,
    )
    _attach_tcp_body(worldbody, tcp_offset)


def build_am_d02_model(urdf_path: str | Path, tcp_offset: np.ndarray) -> mujoco.MjModel:
    """Load the URDF, augment the generated MJCF, and return a MuJoCo model."""

    urdf_path = Path(urdf_path).resolve()
    mesh_dir = urdf_path.parent.parent / "meshes"
    resolved_urdf_path = urdf_path.parent / "_resolved_model.urdf"
    enhanced_xml_path = urdf_path.parent / "_enhanced_scene.xml"

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    mujoco_ext = _ensure_child(root, "mujoco")
    compiler = _ensure_child(mujoco_ext, "compiler")
    compiler.set("meshdir", os.path.relpath(mesh_dir, urdf_path.parent))
    _normalize_mesh_paths(root)

    # First load the URDF into a plain MuJoCo model so we can export valid MJCF.
    tree.write(resolved_urdf_path, encoding="utf-8", xml_declaration=True)
    try:
        basic_model = mujoco.MjModel.from_xml_path(str(resolved_urdf_path))
    finally:
        try:
            resolved_urdf_path.unlink()
        except OSError:
            pass

    # Then reopen the saved MJCF and inject visualization-only helpers.
    try:
        mujoco.mj_saveLastXML(str(enhanced_xml_path), basic_model)
        tree = ET.parse(enhanced_xml_path)
        root = tree.getroot()
        _augment_scene(root, tcp_offset)
        tree.write(enhanced_xml_path, encoding="utf-8", xml_declaration=True)
        return mujoco.MjModel.from_xml_path(str(enhanced_xml_path))
    finally:
        try:
            enhanced_xml_path.unlink()
        except OSError:
            pass
