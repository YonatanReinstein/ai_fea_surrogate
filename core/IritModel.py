import abc
import tempfile
import shutil
import os
import subprocess
import json
from utils.read_inp import read_inp
from pathlib import Path
import platform




class IritModelBase(abc.ABC):
    def __init__(self, irt_script_path: str, dims_json: str = None, dims_dict: dict = None, mesh_threads: int = 1, debug: bool = False, fixed_dims: dict = None):
        if dims_json is None and dims_dict is None:
            raise ValueError("Either dims_json or dims_dict must be provided.")
        os.makedirs("tmp", exist_ok=True)
        if debug:
            self.tmp_dir = "tmp/irit_debug"
            os.makedirs(self.tmp_dir, exist_ok=True)
        else:
            self.tmp_dir = tempfile.mkdtemp(prefix="irit_", dir="tmp")
        self.volume = None
        self._mesh_threads = mesh_threads
        self._debug = debug
        self._fixed_dims = fixed_dims or {}
        self.dims_template = dims_dict
        if self.dims_template is None:
            with open(dims_json, 'r') as f:
                self.dims_template = json.load(f)
        self.__import__script__(Path(irt_script_path))

    def __set_irit_dims__(self):
        with open(f"{self.tmp_dir}/dims.irt", "w") as f:
            f.write("dims = nil();\n")
            for key, value in self.dims_template.items():
                if isinstance(value, dict) and "default" in value:
                    value = value["default"]
                f.write(f"{key} = {value};\n")
                f.write(f"SNOC({key}, dims);\n")
            f.write('save("dims.itd", dims);\n')
            f.write('exit();\n')
        
    def get_volume(self) -> float:
        if self.volume is None:
            if not os.path.exists(f"{self.tmp_dir}/props.txt"):
                self.__exec__script__()
            with open(f"{self.tmp_dir}/props.txt", "r", encoding="utf-8") as f:
                props = {}
                for line in f:
                    if(line.split(" ")[0].strip() == "volume"):
                        props["volume"] = abs(float(line.split(" ")[1].strip()))
                        self.volume = props["volume"]
        return self.volume
    
    def create_mesh(self, U: int =10, V: int =10, W: int =10) -> tuple:
        if not os.path.exists(f"{self.tmp_dir}/model.itd"):
            self.__exec__script__()
        workspace_dir = os.getcwd()

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self._mesh_threads)
        subprocess.run(
            f"irit2inp -s {U} {V} {W} model.itd > model.inp",
            cwd=self.tmp_dir,
            check=True,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        

        #subprocess.run([
        #    "powershell",
        #    "-ExecutionPolicy", "Bypass",
        #    "-File", f"{workspace_dir}/utils/itd_inp_converter.ps1",
        #    "-U", str(U),
        #    "-V", str(V),
        #    "-W", str(W)
        #], cwd=self.tmp_dir, check=True) 
        nodes, elements, elem_to_tile = read_inp(f"{self.tmp_dir}/model.inp")
        return nodes, elements, elem_to_tile

    def get_dim_list(self) -> list:
        dim_list = []
        for key, value in self.dims_template.items():
            if isinstance(value, dict) and "default" in value:
                dim_list.append(value["default"])
            else:
                dim_list.append(value)
        return dim_list

    def __del__(self):
        if self._debug:
            return
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        try:
            os.rmdir("tmp")
        except OSError:
            # Already removed by another instance, or not empty — either is fine.
            pass
    


class IritModel(IritModelBase):    
    def __import__script__(self, irt_script_path: str):
        shutil.copy(irt_script_path, self.tmp_dir)

    def __exec__script__(self):
        self.__set_irit_dims__()
        workspace_dir = os.getcwd()
        subprocess.run([
            "powershell",
            "-ExecutionPolicy", "Bypass",
            "-File", f"{workspace_dir}/utils/irit_script_execution.ps1",
            "-irt_model_path", "model.irt",
        ], cwd=self.tmp_dir, check=True)
    
    
class IritCModel(IritModelBase):
    def __import__script__(self, irt_script_path: str):
        outline_path = irt_script_path.parent / "outline.itd"
        shutil.copy(irt_script_path, self.tmp_dir)
        shutil.copy(outline_path, self.tmp_dir)

    def __set_fixed_dims__(self):
        with open(f"{self.tmp_dir}/fixed_dims.irt", "w") as f:
            f.write("fixed_dims = nil();\n")
            for key, value in self._fixed_dims.items():
                f.write(f"{key} = {value};\n")
                f.write(f"SNOC({key}, fixed_dims);\n")
            f.write('save("fixed_dims.itd", fixed_dims);\n')
            f.write('exit();\n')

    def __exec__script__(self):
        self.__set_irit_dims__()
        system = platform.system()

        irit_cmd = "irit64" if system == "Windows" else "irit"
        model_name = "model.exe" if system == "Windows" else "model"

        subprocess.run(
            [irit_cmd, "-t", "dims.irt"],
            cwd=self.tmp_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if self._fixed_dims:
            self.__set_fixed_dims__()
            subprocess.run(
                [irit_cmd, "-t", "fixed_dims.irt"],
                cwd=self.tmp_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        subprocess.run(
            [f"./{model_name}"],
            cwd=self.tmp_dir,
            check=True
        )




if __name__ == "__main__":
    model = IritCModel("data/bistable/CAD_model/model", "data/bistable/CAD_model/dims.json")
    model.__exec__script__()
    #volume = model.get_volume()
    #print(f"Volume: {volume}")
    nodes, elements = model.create_mesh(U=5, V=5, W=5)
    model
    #print(f"Nodes: {len(nodes)}, Elements: {len(elements)}")


        





        