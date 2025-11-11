import json
import shutil
import os
import subprocess


def str_to_dict(s):
    """Parse 'k=v, k=v' into dict with ints/floats when possible."""
    out = {}
    for part in s.split(","):
        k, v = [p.strip() for p in part.split("=", 1)]
        if v.lstrip("+-").replace(".","",1).isdigit():
            # int or float
            out[k] = float(v) if "." in v else int(v)
        else:
            out[k] = v
    return out

def face_nodes_by_axis(nodes, axis, sign, tol=1e-9):
    from core.node import Node
    idx = {"X":0, "Y":1, "Z":2}[axis]
    coords = [node.coords[idx] for node in nodes]
    target = (max(coords) if sign == "+" else min(coords))
    face = [node for node in nodes if abs(node.coords[idx] - target) <= tol]
    return face

def unit_vector(dir_code):
    return {
        "+X": (1,0,0), "-X": (-1,0,0),
        "+Y": (0,1,0), "-Y": (0,-1,0),
        "+Z": (0,0,1), "-Z": (0,0,-1),
    }[dir_code]

def json_irt_dims_convert(dims_json: str, dims_irt_path: str):
    with open(dims_json, "r") as f:
        dims = json.load(f)

    with open(dims_irt_path, "w") as f:
        for key, value in dims.items():
            if isinstance(value, dict) and "default" in value:
                value = value["default"]
            f.write(f"{key} = {value};\n")


def execute_irit_script(model_irt_path: str, dims_irt_path: str, output_itd_path: str,  collect_props: bool = False):
    tmp_dir = "tmp/irit_execution"
    os.makedirs(tmp_dir, exist_ok=True)
    shutil.copy(model_irt_path, f"{tmp_dir}/model.irt")
    shutil.copy(dims_irt_path, f"{tmp_dir}/dims.irt")
    workspace_dir = os.getcwd()
    subprocess.run([
        "powershell",
        "-ExecutionPolicy", "Bypass",
        "-File", f"{workspace_dir}/utils/irit_script_execution.ps1",
        "-irt_model_path", "model.irt",
    ], cwd=tmp_dir, check=True)
    if collect_props:
        with open(f"{tmp_dir}/props.txt", "r", encoding="utf-8") as f:
            props = {}
            for line in f:
                if(line.split(" ")[0].strip() == "volume"):
                    props["volume"] = abs(float(line.split(" ")[1].strip()))
    if output_itd_path is not None:
        shutil.move(f"{tmp_dir}/model.itd", output_itd_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if collect_props:
        return props
    

def get_inp_from_itd(itd_path: str, output_inp_path: str, U: int =10, V: int =10, W: int =10):
    tmp_dir = "tmp/irit_itd_to_inp"
    os.makedirs(tmp_dir, exist_ok=True)
    shutil.copy(itd_path, f"{tmp_dir}/model.itd")
    workspace_dir = os.getcwd()
    subprocess.run([
        "powershell",
        "-ExecutionPolicy", "Bypass",
        "-File", f"{workspace_dir}/utils/itd_inp_converter.ps1",
        "-U", str(U),
        "-V", str(V),
        "-W", str(W)
    ], cwd=tmp_dir, check=True)
    shutil.move(f"{tmp_dir}/model.inp", output_inp_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)



if __name__ == "__main__":
    #test the execute_irit_script function
    props = execute_irit_script(
        model_irt_path="data/beam/CAD_model/model.irt",
        dims_irt_path="tmp/dims.irt",
        output_itd_path="tmp/model.itd",
        collect_props=True
    )
    print(props["volume"])


