import torch
from abc import ABC
from utils.gnn_surrogate import GNN
from evaluators.base_evaluator import BaseEvaluator
from core.IritModel import IIritModel
from core.component import Component
from training.gnn_training import gnn_input_fn, gnn_target_fn
import json
from multiprocessing import Pool
from torch_geometric.loader import DataLoader
import importlib
import threading
from time import sleep
import socket

STOP = False   

def listener():
    global STOP
    s = socket.socket()
    s.bind(("127.0.0.1", 5001))
    s.listen(1)
    conn, addr = s.accept()
    msg = conn.recv(16)
    if msg == b"STOP":
        STOP = True
    conn.close()
    s.close()





def _run_sample_worker(args):
    (
        model_path,
        dims,
        young,
        poisson,
        sample_index,
        screenshot,
        anchor_condition,
        force_pattern,
        U,
        V,
        W
    ) = args
    import random
    sleep(random.uniform(0.1, 0.2))  # Simulate variable computation time
    #print (f"Worker started for sample index: {sample_index%200}")
    cad_model = IIritModel(model_path, dims_dict=dims)
    component = Component(cad_model, young, poisson)
    component.generate_mesh(U=U, V=V, W=W)
    component.mesh.anchor_nodes_by_condition(anchor_condition)
    component.mesh.apply_force_by_pattern(force_pattern)
    data = component.to_graph_with_labels(with_labels=False)  
    if screenshot:
        save_path=f"screenshots/mesh_{sample_index+1}.png"
        component.mesh.plot_mesh(save_path=save_path)
    volume = component.get_volume()
    return data, volume


class GNNEvaluator(BaseEvaluator):
    def __init__(self, geometry_name: str,screenshots: bool = False, processes: int = None):
        super().__init__(geometry_name)
        self.processes = processes
        self.screenshots = screenshots

        ckpt_path = f"data/{geometry_name}/gnn_surrogate.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.target_mean = ckpt["target_mean"].to(self.device)
        self.target_std  = ckpt["target_std"].to(self.device)
        self.x_mean = ckpt["x_mean"].to(self.device)
        self.x_std  = ckpt["x_std"].to(self.device)

        self.node_in_dim = ckpt["node_in_dim"]

        self.model = GNN(
            node_in_dim=self.node_in_dim,
            hidden_dim=128,
            num_layers=6
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.sample_counter = 0

        material_properties_path = f"data/{self.geometry_name}/CAD_model/material_properties.json"
        material_properties = json.loads(open(material_properties_path, "r").read())
        self.young = material_properties["young_modulus"]
        self.poisson = material_properties["poisson_ratio"]
        self.yield_strength = material_properties["yield_strength"]
        self.model_path = f"data/{self.geometry_name}/CAD_model/model.irt"

        module = importlib.import_module(f"data.{geometry_name}.boundary_conditions")
        self.anchor_condition = module.anchor_condition
        self.force_pattern = module.force_pattern
        self.mesh_resolution = module.mesh_resolution
        self.U, self.V, self.W = self.mesh_resolution()
        self.listener_thread = threading.Thread(target=listener, daemon=True).start()



    def evaluate(self, dims_list: list[dict]):
        #print("start evaluation of batch size:", len(dims_list))
        batch_size = len(dims_list)


        # Unique screenshot indexes
        indexes = list(range(self.sample_counter, self.sample_counter + batch_size))
        self.sample_counter += batch_size

        # Build args
        all_args = [
            (
             self.model_path,
             dims, 
             self.young,
             self.poisson,
             idx , 
             self.screenshots,
             self.anchor_condition,
             self.force_pattern,
             self.U,
             self.V,
             self.W)
            for dims, idx in zip(dims_list, indexes)
        ]
        results = []


        if self.processes is None:
            pool = Pool()
        else:
            pool = Pool(processes=self.processes)
        it = pool.imap(_run_sample_worker, all_args)
        try:
            for result in it:
                if STOP:
                    print("Graceful stop requested.")

                    # -------- KEY PART --------
                    pool.close()  
                    sleep(30)      # do not accept new tasks
                    pool.terminate()    # kill worker process
                    # -------------------------
                    print("Stopped during evaluation.")
                    pool.join() 

                    break

                results.append(result)
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Terminating pool.")
            pool.terminate()

        #print("pool completed.")

        #print("Preparing GNN inputs...")




        graph_list, volume_list = zip(*results)

        # Build DataLoader
        batch_size = int(len(dims_list) / 8)
        all_stress = []

        loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False)
        for batch_data in loader:
            # Prepare inputs
            x, edge_index, batch = gnn_input_fn(batch_data)
            x[:, 3] = x[:, 3] / 1e+6  
            torch.set_printoptions(threshold=torch.inf)
            
            x = x.to(self.device)

            x = (x - self.x_mean) / self.x_std
        
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long)

            edge_index = edge_index.to(self.device)
            batch = batch.to(self.device)

            # Predict
            with torch.no_grad():
                pred_norm = self.model(x, edge_index, batch)

            # Denormalize
            stress = pred_norm * self.target_std + self.target_mean
            stress = stress.squeeze()
            #print("Evaluation completed.")

            all_stress.extend(stress.detach().cpu().numpy().tolist())
            

        return {
            "stress": all_stress,        # shape [batch_size]
            "volume": volume_list,    # list of floats
            "yield_strength": self.yield_strength
        }



if __name__ == "__main__":
    evaluator = GNNEvaluator("arm")
    dims_example = {
        "d1": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d2": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d3": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d4": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d5": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d6": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d7": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d8": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d9": {"default": 3.0, "min": 0.0, "max": 3.0},
        "d10": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d11": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d12": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d13": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d14": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d15": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d16": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d17": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d18": {"default": 3.0, "min": 0.0, "max": 3.0},
        "d19": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d20": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d21": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d22": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d23": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d24": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d25": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d26": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d27": {"default": 3.0, "min": 0.0, "max": 3.0},
        "d28": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d29": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d30": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d31": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d32": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d33": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d34": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d35": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d36": {"default": 3.0, "min": 0.0, "max": 3.0}
    }
    results = evaluator.evaluate([dims_example])#, dims_example])
    print(results)