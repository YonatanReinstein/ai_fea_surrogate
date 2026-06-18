"""
Generate outline.irt from fixed_dims() then compile to outline.itd.
Run from the project root: python data/hollow_cube/CAD_model/gen_outline.py
"""
import subprocess
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))
from data.hollow_cube.boundary_conditions import fixed_dims

CAD_DIR = Path(__file__).parent

def write_irt(length: float, height: float, width: float) -> None:
    irt = CAD_DIR / "outline.irt"
    irt.write_text(
        f"length = {length};\n"
        f"height = {height};\n"
        f"width = {width};\n"
        "\n\n"
        "TV = GBOX(\n"
        "    point(0,0,0),\n"
        "    point(0 , 0 , length),\n"
        "    point(0 , height , 0),\n"
        "    point(0,height , length),\n"
        "    point(width, 0 , 0),\n"
        "    point(width, 0 , length),\n"
        "    point(width, height , 0),\n"
        "    point(width, height , length)\n"
        ");\n"
        'save("outline.itd", list(TV));\n'
        "free(TV);\n"
        "exit();\n"
    )

if __name__ == "__main__":
    fd = fixed_dims()
    write_irt(length=fd["d1"], height=fd["d2"], width=fd["d3"])

    irit_cmd = "irit64" if platform.system() == "Windows" else "irit"
    subprocess.run([irit_cmd, "-t", "outline.irt"], cwd=CAD_DIR, check=True)
    print(f"Generated {CAD_DIR / 'outline.itd'}")
