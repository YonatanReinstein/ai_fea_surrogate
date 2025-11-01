import ansys.mapdl.core as pymapdl
mapdl = pymapdl.launch_mapdl()

mapdl.prep7()
mapdl.et(1, "SOLID185")
mapdl.cdread("db", "beam", ext="inp")  # if imported from your irit2inp file
mapdl.eplot(show_edges=True)
mapdl.nplot(show_node_numbering=True)

print(mapdl.get('ncount', 'node', 0, 'count'))
print(mapdl.get('ecount', 'elem', 0, 'count'))

