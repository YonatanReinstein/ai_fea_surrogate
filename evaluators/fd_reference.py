"""Shared mesh/solve helpers for the remesh finite-difference ground truth.

The FD reference -- perturb one design variable, regenerate the geometry from
the CAD model, re-solve, difference the resulting ks_stress -- is what
evaluators/validate_axis_gradient.py and evaluators/validate_node_gradient.py
measure the analytic gradient against. Both need to build a solvable Mesh from
raw node/element dicts and to run a whole design point end to end, so those two
steps live here rather than in either script.
"""
from core.IritModel import IritCModel
from core.mesh import Mesh
from core.node import Node
from core.element import Element


def _build_mesh(nodes_dict, elements_dict, anchor_condition, force_pattern,
                load_scale=1.0, total_force=None):
    nodes = {nid: Node(nid, list(xyz)) for nid, xyz in nodes_dict.items()}
    elements = {eid: Element(eid, [nodes[nid] for nid in nlist])
                for eid, nlist in elements_dict.items()}
    mesh = Mesh(nodes, elements)
    mesh.anchor_nodes_by_condition(anchor_condition)
    mesh.apply_force_by_pattern(force_pattern)
    if total_force is not None:
        # This repo's boundary_conditions.py files divide a fixed resultant
        # by a node count taken from the geometry's own mesh_resolution(),
        # so overriding the resolution silently changes the TOTAL applied
        # load -- which makes a mesh-refinement study compare different
        # physical problems. Renormalize the resultant to a fixed value so
        # only the discretization varies.
        cur = sum(n.forces[1] for n in mesh.all_nodes())
        if cur != 0.0:
            k = total_force / cur
            for n in mesh.all_nodes():
                n.forces = [f * k for f in n.forces]
    if load_scale != 1.0:
        for n in mesh.all_nodes():
            n.forces = [f * load_scale for f in n.forces]
    return mesh


def _solve_ks(exe_path, dims, U, V, W, fixed_dims, mesh_threads,
              anchor_condition, force_pattern, young, poisson,
              ccx_path, nproc, rho_mpa, load_scale=1.0, total_force=None):
    """Full pipeline for one design: CAD -> mesh -> ccx -> ks_stress.

    Deliberately mirrors compute_thickness_jacobian's own mesh generation
    (IritCModel.create_mesh directly, not Component.generate_mesh) so the FD
    ground truth and the analytic gradient see the same geometry pipeline.
    """
    model = IritCModel(exe_path, dims_dict=dims, fixed_dims=fixed_dims,
                       mesh_threads=mesh_threads)
    nodes_dict, elements_dict, _ = model.create_mesh(U=U, V=V, W=W)
    mesh = _build_mesh(nodes_dict, elements_dict, anchor_condition, force_pattern,
                       load_scale=load_scale, total_force=total_force)
    mesh.solve_ccx(young, poisson, ccx_path=ccx_path, nproc=nproc)
    return mesh.get_aggregated_stress(rho=rho_mpa * 1e-6)
