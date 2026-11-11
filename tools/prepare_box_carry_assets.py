"""Copy the existing OmniContact G1 and its referenced meshes into this project's ignored data directory."""

import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=root.parent / "omnicontact_sim2sim" / "g1_description")
    args = ap.parse_args()
    source = args.source / "g1_29dof.xml"
    target = root / "data" / "g1_carry"
    xml = ET.parse(source)
    meshdir = xml.find("compiler").get("meshdir", "")
    (target / meshdir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target / source.name)
    meshes = xml.findall("asset/mesh")
    for mesh in meshes:
        file = Path(mesh.get("file"))
        dest = target / meshdir / file
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source / meshdir / file, dest)
    print(f"Prepared {target / source.name} with {len(meshes)} meshes")


if __name__ == "__main__":
    main()
