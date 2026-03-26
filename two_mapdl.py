from ansys.mapdl.core import launch_mapdl

m1 = launch_mapdl(mode="grpc", nproc=1, override=True)
print("m1 ok")
m2 = launch_mapdl(mode="grpc", nproc=1, override=True)
print("m2 ok")
m3 = launch_mapdl(mode="grpc", nproc=1, override=True)
print("m3 ok")
m4 = launch_mapdl(mode="grpc", nproc=1, override=True)
print("m4 ok")


m1.exit()
m2.exit()
m3.exit()
m4.exit()