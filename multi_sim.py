from ansys.mapdl.core import MapdlPool
import numpy as np

def run_cube(mapdl, load):
    mapdl.clear()
    mapdl.prep7()

    # --- material ---
    mapdl.et(1, 185)
    mapdl.mp("EX", 1, 2e11)
    mapdl.mp("PRXY", 1, 0.3)

    # --- geometry ---
    mapdl.block(0, 1, 0, 1, 0, 1)

    # --- mesh ---
    mapdl.esize(0.25)
    mapdl.vmesh("ALL")

    # --- anchor face x=0 ---
    mapdl.nsel("S", "LOC", "X", 0)
    mapdl.d("ALL", "ALL", 0)

    # --- load face x=1 ---
    mapdl.nsel("S", "LOC", "X", 1)

    # distribute load equally
    n_nodes = mapdl.get("NN", "NODE", 0, "COUNT")
    f_per_node = load / n_nodes

    mapdl.f("ALL", "FX", f_per_node)

    mapdl.allsel()

    # --- solve ---
    mapdl.run("/SOLU")
    mapdl.antype("STATIC")
    mapdl.solve()

    # --- post ---
    mapdl.post1()
    mapdl.set("LAST")

    stress = mapdl.post_processing.nodal_eqv_stress()

    return float(np.max(stress))


if __name__ == "__main__":
    # --- create pool ---
    pool = MapdlPool(n_instances=1, nproc=1)  # try 2–4

    loads = [1e5, 2e5, 3e5, 4e5]# 5e5, 6e5]

    results = pool.map(run_cube, loads)

    print("Results:", results)

    pool.exit()