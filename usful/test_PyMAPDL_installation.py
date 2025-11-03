from ansys.mapdl.core import launch_mapdl

mapdl = launch_mapdl(run_location='.', mode='grpc', override=True)
#print(mapdl)
#mapdl.exit()
print(mapdl.shpp("SUMM"))
print(mapdl.lswrite())
print(mapdl.sfelist())
