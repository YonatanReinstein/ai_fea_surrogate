import json
from utils.models import Mesh  
from typing import Dict
import subprocess
import os





class Component:
    def __init__(self, irit_model_name: str):
        self.irit_path = "data/" + irit_model_name + ".irt"  
        self.model_name = irit_model_name
        self.dims = None  # design dimensions
        self.mesh = None  # placeholder for the mesh object
        self.volume = None  # placeholder for volume data
    
    def generate_mesh(self):
        with open("data/" + self.model_name + ".json", "r") as f:
            self.dims = json.load(f)
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/dims.irt", "w") as f:
            for key, value in self.dims.items():
                f.write(f"{key} = {value};\n")
        subprocess.run([
            "powershell",
            "-ExecutionPolicy", "Bypass",
            "-File", "utils/irit2inp.ps1",
            "-irtfile", self.model_name
        ])
        self.mesh = Mesh.from_inp("tmp/" + self.model_name + ".inp")
        with open("tmp/" + self.model_name + "_out.txt", "r") as f:
            lines = f.readlines()

            for line in lines:
                if(line.split(" ")[0].strip() == "volume"):
                    self.volume = abs(float(line.split(" ")[1].strip()))
                    break
            
if __name__ == "__main__":
    component = Component("beam")
    component.generate_mesh()