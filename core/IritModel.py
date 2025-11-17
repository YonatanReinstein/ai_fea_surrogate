import tempfile
import shutil
import os
import subprocess
import json
from utils.read_inp import read_inp


class IIritModel:
    def __init__(self, irt_script_path: str, dims_json: str = None, dims_dict: dict = None):
        if dims_json is None and dims_dict is None:
            raise ValueError("Either dims_json or dims_dict must be provided.")
        os.makedirs("tmp", exist_ok=True)
        self.tmp_dir = tempfile.mkdtemp(prefix="irit_", dir="tmp", )
        self.volume = None
        self.dims_template = dims_dict
        shutil.copy(irt_script_path, os.path.join(self.tmp_dir, "model.irt"))
        if dims_dict is None:
            with open(dims_json, 'r') as f:
                self.dims_template = json.load(f)
        
    def __exec__script__(self):
        self.__set_irit_dims__()
        workspace_dir = os.getcwd()
        subprocess.run([
            "powershell",
            "-ExecutionPolicy", "Bypass",
            "-File", f"{workspace_dir}/utils/irit_script_execution.ps1",
            "-irt_model_path", "model.irt",
        ], cwd=self.tmp_dir, check=True)
        with open(f"{self.tmp_dir}/props.txt", "r", encoding="utf-8") as f:
            props = {}
            for line in f:
                if(line.split(" ")[0].strip() == "volume"):
                    props["volume"] = abs(float(line.split(" ")[1].strip()))
        self.volume = props["volume"]
        
    def get_volume(self) -> float:
        if self.volume is None:
            self.__exec__script__()
        return self.volume
    
    def create_mesh(self, U: int =10, V: int =10, W: int =10) -> tuple:
        if not os.path.exists(f"{self.tmp_dir}/model.itd"):
            self.__exec__script__()
        workspace_dir = os.getcwd()
        subprocess.run([
            "powershell",
            "-ExecutionPolicy", "Bypass",
            "-File", f"{workspace_dir}/utils/itd_inp_converter.ps1",
            "-U", str(U),
            "-V", str(V),
            "-W", str(W)
        ], cwd=self.tmp_dir, check=True)
        nodes, elements = read_inp(f"{self.tmp_dir}/model.inp")
        return nodes, elements

    def __set_irit_dims__(self):
        with open(f"{self.tmp_dir}/dims.irt", "w") as f:
            for key, value in self.dims_template.items():
                if isinstance(value, dict) and "default" in value:
                    value = value["default"]
                f.write(f"{key} = {value};\n")

    def __del__(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        if os.path.exists("tmp") and len(os.listdir("tmp")) == 0:
            os.rmdir("tmp")
    
    def get_dim_list(self) -> list:
        dim_list = []
        for key, value in self.dims_template.items():
            if isinstance(value, dict) and "default" in value:
                dim_list.append(value["default"])
            else:
                dim_list.append(value)
        return dim_list


if __name__ == "__main__":
    model = IIritModel("data/beam/CAD_model/model.irt", "data/beam/CAD_model/dims.json")
    volume = model.get_volume()
    print(f"Volume: {volume}")
    nodes, elements = model.create_mesh(U=5, V=5, W=5)
    print(f"Nodes: {len(nodes)}, Elements: {len(elements)}")


        





        