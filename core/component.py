import argparse
from .mesh import Mesh  
from utils.data_processing import json_irt_dims_convert, execute_irit_script, get_inp_from_itd
import os
import shutil


class Component:
    def __init__(self, irt_script_path: str, dims_path: str, young: float = 2.1e11, poisson: float = 0.3):
        self.model_path = irt_script_path
        self.dims_path = dims_path
        self.dims = None
        self.mesh = None
        self.young = young
        self.poisson = poisson

    def clear_boundaries(self):
        if self.mesh is not None:
            self.mesh.clear_anchors()
            self.mesh.clear_forces()

    def generate_mesh(self, dims_path: str = None):
        tmp_dir = "tmp/mesh_generation"
        if dims_path is None:
            dims_path = self.dims_path
        os.makedirs(tmp_dir, exist_ok=True)
        shutil.copy(self.model_path, f"{tmp_dir}/model.irt")
        dims_irt_path = f"{tmp_dir}/dims.irt"
        json_irt_dims_convert(dims_path, dims_irt_path)
        execute_irit_script(model_irt_path=f"{tmp_dir}/model.irt", dims_irt_path=dims_irt_path, output_itd_path=f"{tmp_dir}/model.itd")
        inp_path = f"{tmp_dir}/model.inp"
        get_inp_from_itd(itd_path=f"{tmp_dir}/model.itd", output_inp_path=inp_path)
        self.mesh = Mesh.from_inp(inp_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    def ansys_sim(self):
        if self.mesh is None:
            raise ValueError("Mesh has not been generated yet.")
        self.mesh.solve(self.young, self.poisson) 
        return self.mesh.get_max_stress(), self.mesh.get_max_displacement()

    def get_volume(self):
        props = execute_irit_script(
            model_irt_path=self.model_path,
            dims_irt_path=self.dims_path,
            output_itd_path=None,
            collect_props=True
        )
        return props["volume"]



if __name__ == "__main__":

    import argparse
    from utils.data_processing import str_to_dict

    parser = argparse.ArgumentParser()
    parser.add_argument("--young", type=float, default=2.1e11, help="Young's modulus [Pa]")
    parser.add_argument("--poisson", type=float, default=0.3, help="Poisson ratio [-]")
    # Repeatable flags:
    parser.add_argument("--anchor", action="append",default=["cube=1,face=-X"])
    parser.add_argument("--force", action="append",default=["cube=9,face=+X,type=nodal,value=1e6"])

    args = parser.parse_args()
    model_path = "data/arm/CAD_model/model.irt"
    json_path = "data/arm/CAD_model/dims.json"
    component = Component(model_path, json_path)
    component.generate_mesh()
    i = 0
    j = 0
    for node in component.mesh.nodes.values():
        if abs(node.coords[2]) < 1e-6:
            node.anchored = True
            i += 1
            print(f"Anchored node ID: {node.id}, Coords: {node.coords}")
    print(f"Total anchored nodes: {i}")

    for node in component.mesh.nodes.values():
        if abs(node.coords[2] - 9) < 1e-6:
            node.forces = [1e6, 0.0, 0.0]
            j += 1
            print(f"Applying force to node ID: {node.id}, Coords: {node.coords}")
    print(f"Total force nodes: {j}")

    #for a in args.anchor:
    #    spec = str_to_dict(a)
    #    eid = int(spec["cube"])  # element ID provided by user
    #    face = spec["face"].upper()
    #    component.mesh.add_anchor(eid, face)

    #for f in args.force:
    #    print(f"Applying force: {f}")
    #    spec = str_to_dict(f)
    #    eid = int(spec["cube"])
    #    face = spec["face"].upper()
    #    value = float(spec["value"])
    #    dir_code = str(spec.get("dir", face)).upper()
    #    component.mesh.add_force(eid, face, value, dir_code)

    component.mesh.solve(args.young, args.poisson)


