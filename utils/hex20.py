"""Second-order (20-node) hex meshes, built from irit2inp's linear output.

irit2inp only ever emits 8-node bricks (`*ELEMENT,TYPE=C3D8`), and it is the
only mesher in this pipeline, so a quadratic element cannot be asked of the
CAD side. It does not have to be: a C3D20 is a C3D8 with a node added at the
midpoint of each of its 12 edges, and on a mesh whose faces are already
straight-edged (which irit2inp's is -- it samples the tile surface at the
U/V/W grid and joins the samples with straight element edges) placing that
node at the exact edge midpoint adds no geometric error at all. The element
geometry is unchanged; only the DISPLACEMENT interpolation goes from tri-
linear to quadratic.

WHY BOTHER. The stress this project optimises is carried by bending of a thin
tube wall, and a trilinear hex cannot represent bending: it needs several
elements through the wall before it stops locking, which is the whole reason
`mesh_resolution` keeps being pushed up (3,3,3 -> 5,3,3) at a cost that is
cubic in wall resolution. One quadratic element spans a linear strain
gradient exactly, so two of them through a wall (mesh_resolution (3,3,3)) is
a genuinely bending-capable discretisation.

TWO CONVENTIONS COLLAPSE INTO ONE. The linear path has to run the reported
stress on C3D8I and the gradient on plain C3D8 -- C3D8I to beat shear locking
in the value, C3D8 because ccx's semi-analytical sensitivity differentiates
C3D8I's statically condensed incompatible modes badly (direction cosine -0.68
vs +0.90; see Mesh.solve_ccx_axis_gradient). At quadratic order there are no
incompatible modes to add or to differentiate, so g0 and dG/dx come from the
same element for the first time.

ccx SUPPORT is first class, not incidental: getdesiinfo3d.f branches on
nope=20 explicitly (with its own 8-node face table and nopedesi=5) and
accepts exactly nope 8 and 20 for hex design variables, and e_c3d_se.f
carries a '20' branch. Nothing in tools/build_ccx_axis.sh's three patches
looks at the element type.

C3D20 vs C3D20R is the integration rule: 27 Gauss points against 8. C3D20R is
several times cheaper and is CalculiX's own recommendation for most work;
C3D20 is the safer default here because *SENSITIVITY differentiates whatever
the element does, and reduced integration adds hourglass-control behaviour
that has not been checked against the FD reference on this geometry.
"""
import os

# Node order of a CalculiX/Abaqus C3D20: corners 1-8 exactly as the C3D8 they
# come from, then the four bottom-face edge midpoints, the four top-face edge
# midpoints, and finally the four vertical edges -- local nodes 9-20 in that
# order. Written 0-based as (corner, corner) pairs.
HEX20_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),   # local 9-12,  bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),   # local 13-16, top face
    (0, 4), (1, 5), (2, 6), (3, 7),   # local 17-20, verticals
)

QUADRATIC_TYPES = ("C3D20", "C3D20R")


def is_quadratic(element_type: str) -> bool:
    return str(element_type).strip().upper() in QUADRATIC_TYPES


def requested_element_type(default: str = "C3D8I") -> str:
    """The element the whole pipeline should use, from FEA_ELEMENT_TYPE.

    Read from the environment for the same reason mesh_resolution() reads
    FEA_MESH_RES: the mesh is built inside ProcessPoolExecutor workers, where
    a monkeypatch in the parent is invisible but an inherited env var is.
    """
    return (os.environ.get("FEA_ELEMENT_TYPE") or default).strip().upper()


def quadratic_requested() -> bool:
    return is_quadratic(requested_element_type())


def quadratic_element_type() -> str:
    """The 20-node type name to write into the deck (C3D20 unless the
    environment asked for the reduced-integration variant)."""
    elt = requested_element_type()
    return elt if is_quadratic(elt) else "C3D20"


def hex8_to_hex20(nodes: dict, elements: dict):
    """(nodes, elements) of 8-node hexes -> the same mesh as 20-node hexes.

    Every element keeps its id and its first eight node ids, so anything
    keyed on element id -- elem_to_tile above all -- stays valid untouched.
    New nodes are appended above the highest existing id, contiguously, which
    is what the `*NSET,NSET=NALL,GENERATE 1,max` card in Mesh's deck assumes.

    Deterministic in the connectivity alone (elements are visited in sorted
    id order and an edge is keyed by its sorted endpoint pair), so two
    meshes of the same design with different tile thicknesses get identical
    node numbering -- which is what lets compute_thickness_jacobian keep
    differencing coordinates by node id. A midside node sits at the mean of
    its two corners, so its dX/dt comes out as the mean of theirs, exactly
    as it should.
    """
    nodes = dict(nodes)
    elements = dict(elements)
    next_id = max(nodes) + 1
    midside = {}

    for eid in sorted(elements):
        corners = list(elements[eid])[:8]
        if len(corners) != 8:
            raise ValueError(
                f"element {eid} has {len(elements[eid])} nodes; hex8_to_hex20 "
                "needs an 8-node hex mesh")
        extra = []
        for a, b in HEX20_EDGES:
            na, nb = corners[a], corners[b]
            key = (na, nb) if na < nb else (nb, na)
            mid = midside.get(key)
            if mid is None:
                xa, xb = nodes[na], nodes[nb]
                mid = next_id
                next_id += 1
                nodes[mid] = tuple((xa[k] + xb[k]) / 2.0 for k in range(3))
                midside[key] = mid
            extra.append(mid)
        elements[eid] = corners + extra

    return nodes, elements
