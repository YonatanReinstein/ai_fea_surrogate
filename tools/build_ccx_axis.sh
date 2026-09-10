#!/usr/bin/env bash
# Build a patched CalculiX 2.17 ("ccx_axis") that can report the stress
# sensitivity along the GLOBAL X/Y/Z AXES instead of along each design node's
# surface normal.
#
# WHY
#   Stock ccx's *DESIGNVARIABLES,TYPE=COORDINATE gives one scalar per design
#   node -- d(objective)/d(displacement along NORM), where NORM is the mean of
#   the adjacent external face normals -- divided by the node's tributary area.
#   At an edge or corner NORM is a bisector pointing somewhere with no relation
#   to the tile parameter we differentiate against, and the direction cannot be
#   chosen: *FILTER,EDGEPRESERVATION only restricts WHICH faces vote, so NORM
#   always stays a combination of face normals.
#
#   In sensitivity.c the surface normal enters the computation through exactly
#   one array, xdesi (extnor is freed immediately after xdesi is filled), so
#   overwriting xdesi with a global axis makes the reported DFDN the derivative
#   along that axis. Three runs give the full discrete gradient dKS/dX[n][j].
#
# PATCHES (all three are no-ops unless their environment variable is set, so
# the resulting binary is bit-identical to stock ccx when none are -- verified
# against the distro ccx on hollow_cube_one, max |diff| = 0.0)
#
#   CCX_DESI_AXIS=1|2|3   sensitivity.c: replace every design direction with
#                         the global x/y/z axis instead of the surface normal.
#   CCX_NO_AREA_NORM=1    distributesens.f: skip the division by weightformgrad
#                         (the nodal tributary area), so DFDN is a plain
#                         derivative rather than a per-unit-area density.
#   CCX_DESI_STEP=<len>   sensitivity.c: override distmin, the smallest element
#                         edge length, which is the step of the forward
#                         difference ccx takes on the element stiffness.
#   CCX_DESI_INTERIOR=1   getdesiinfo3d.f: keep INTERIOR nodes as design
#                         variables instead of deleting them. Stock ccx drops
#                         any node not on an external face because a SHAPE
#                         sensitivity needs a surface normal -- with
#                         CCX_DESI_AXIS there is no normal, so the restriction
#                         is meaningless, and interior nodes turn out to carry
#                         the single largest error: on hollow_cube_one/d1 at
#                         mesh 5,5,5 they are 16.6% of surface magnitude and
#                         contribute -9.0% of the tile derivative (opposite
#                         sign, so they cancel part of the surface sum).
#                         Contracting over surface+interior reproduces the
#                         remesh FD to 0.12%; surface only is +9.1% off.
#
# VALIDATION (evaluators/validate_axis_gradient.py)
#   MASS objective, hollow_cube_one -- purely geometric, no adjoint, so it
#   tests this plumbing alone: 46/46 nodes match an exact python volume
#   gradient with direction cosine 1.00000000 and magnitude ratio 1.00000000
#   (worst componentwise relative error 1.1e-8).
#   STRESS objective, hollow_cube_mid at 5,5,5 with C3D8, 48 tiles contracted
#   against dX/dt and compared to remesh FD, over the 26 tiles whose |FD| is
#   above 0.1% of max (the rest are FD noise) -- per-tile ratio to remesh FD:
#     stock normal projection : median 1.007  min 0.084  max 2.645  spread 31.4x
#     axis, surface only      : median 1.036  min 0.634  max 1.530  spread 2.41x
#     axis + CCX_DESI_INTERIOR: median 1.000  min 0.981  max 1.094  spread 1.12x
#   All three at 3 solves per design regardless of tile count.
#
# REQUIREMENTS
#   gcc, make, perl, curl, and an existing CalculiX runtime (libarpack,
#   liblapack, libblas, libatlas) -- this repo's is under ~/opt/calculix.
#   gfortran and the SPOOLES headers are fetched as .deb files and unpacked
#   locally; no root is needed.
set -euo pipefail

PREFIX="${PREFIX:-$HOME/opt/calculix}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/ccx_axis_build}"
CCXLIB="$PREFIX/root/usr/lib/x86_64-linux-gnu"
OUT="$PREFIX/root/usr/bin/ccx_axis"
[ -d "$CCXLIB" ] || { echo "no CalculiX runtime libs at $CCXLIB" >&2; exit 1; }

mkdir -p "$BUILD"/{debs,toolroot,linkdir}
cd "$BUILD"

echo "==> fetching CalculiX 2.17 source"
[ -f ccx_2.17.src.tar.bz2 ] || curl -sSf -O https://www.dhondt.de/ccx_2.17.src.tar.bz2
[ -d CalculiX ] || tar xjf ccx_2.17.src.tar.bz2

echo "==> fetching gfortran + SPOOLES headers (no root)"
if [ ! -x toolroot/usr/bin/gfortran-11 ]; then
  ( cd debs && apt-get download gfortran-11 libgfortran-11-dev libspooles-dev )
  for d in debs/*.deb; do dpkg-deb -x "$d" toolroot; done
fi
# The gfortran-11 deb ships f951 but not the shared gcc support files (lto
# plugin, collect2, crt*.o); borrow them from the system gcc of the same major.
GCCLIB=/usr/lib/gcc/x86_64-linux-gnu/11
TOOLGCC="$BUILD/toolroot/usr/lib/gcc/x86_64-linux-gnu/11"
for f in "$GCCLIB"/*; do
  [ -e "$TOOLGCC/$(basename "$f")" ] || ln -s "$f" "$TOOLGCC/$(basename "$f")"
done

# ccx links -lfoo, the runtime bundle only has libfoo.so.N -- provide the names.
ln -sf "$CCXLIB/libarpack.so.2" linkdir/libarpack.so
ln -sf "$CCXLIB/liblapack.so.3" linkdir/liblapack.so
ln -sf "$CCXLIB/libblas.so.3"   linkdir/libblas.so
ln -sf "$CCXLIB/libatlas.so.3"  linkdir/libatlas.so

SRC="$BUILD/CalculiX/ccx_2.17/src"

echo "==> patching"
python3 - "$SRC" <<'PYEOF'
import sys
S = sys.argv[1]

p = S + "/sensitivity.c"
src = open(p).read()
if "CCX_DESI_AXIS" in src:
    print("   sensitivity.c already patched")
else:
    anchor = "    FORTRAN(smalldist,(co,&distmin,lakon,ipkon,kon,ne));\n"
    assert src.count(anchor) == 1
    src = src.replace(anchor, anchor + """
    /* ---- CCX_DESI_STEP patch --------------------------------------------
       distmin is the smallest element edge length, and it is the step of the
       forward difference CalculiX takes on the element stiffness to build the
       semi-analytical sensitivity (see mafillsmmain_se / stress_sen_dx.c,
       where dgdx = dksper/distmin).  CCX_DESI_STEP overrides it with an
       absolute length.  Unset => stock CalculiX behaviour.
       ---------------------------------------------------------------------- */
    {
      char *ccxstep=getenv("CCX_DESI_STEP");
      if(ccxstep!=NULL){
        double stepval=atof(ccxstep);
        if(stepval>0.){
          printf(" CCX_DESI_STEP: perturbation step %e replaces the smallest\\n"
                 " element edge length %e.\\n\\n",stepval,distmin);
          distmin=stepval;
        }else{
          printf(" *WARNING: CCX_DESI_STEP=\\"%s\\" is not positive; ignored.\\n\\n",
                 ccxstep);
        }
      }
    }
""")
    anchor2 = "    for(k=0;k<3*ndesi;k++){\n      xdesi[k]*=distmin;\n    }\n"
    assert src.count(anchor2) == 1
    src = src.replace(anchor2, anchor2 + """
    /* ---- CCX_DESI_AXIS patch -------------------------------------------
       If CCX_DESI_AXIS is set to 1, 2 or 3, the design direction of every
       design variable is replaced by the global x-, y- or z-axis instead of
       the outward surface normal.  DFDN in the .frd is then the derivative of
       the objective w.r.t. motion of that node along that global axis.  The
       normal reaches the computation only through xdesi (extnor is freed a few
       lines below), so this is the whole change.  Unset => stock CalculiX.
       -------------------------------------------------------------------- */
    {
      char *ccxaxis=getenv("CCX_DESI_AXIS");
      if(ccxaxis!=NULL){
        ITG iaxis=atoi(ccxaxis)-1;
        if((iaxis>=0)&&(iaxis<=2)){
          for(k=0;k<ndesi;k++){
            xdesi[3*k]=0.;xdesi[3*k+1]=0.;xdesi[3*k+2]=0.;
            xdesi[3*k+iaxis]=distmin;
          }
          printf(" CCX_DESI_AXIS=%d: design directions overridden to the\\n"
                 " global %c-axis for all %d design variables.\\n\\n",
                 iaxis+1,'x'+iaxis,(int)ndesi);
        }else{
          printf(" *WARNING: CCX_DESI_AXIS=\\"%s\\" is not 1, 2 or 3;\\n"
                 "          surface normals are used.\\n\\n",ccxaxis);
        }
      }
    }
""")
    open(p, "w").write(src)
    print("   patched sensitivity.c")

p = S + "/distributesens.f"
src = open(p).read()
if "CCX_NO_AREA_NORM" in src:
    print("   distributesens.f already patched")
else:
    d = "     &   nopedesi,nnodes,nodedesiinv(*),nobject,iregion,iactnode\n"
    assert src.count(d) == 1
    src = src.replace(d, d + "!\n      character*132 cnoarea\n")
    s = "!     \n!     Scaling of sensitivities\n!     \n      do idesvar=1,ndesi\n"
    assert src.count(s) == 1
    src = src.replace(s, """!     
!     Scaling of sensitivities
!
!     ---- CCX_NO_AREA_NORM patch ----------------------------------------
!     Stock CalculiX divides the raw derivative by weightformgrad (the
!     nodal tributary area) so that DFDN is a per-unit-area density.  With
!     CCX_NO_AREA_NORM=1 that division is skipped and DFDN is the plain
!     derivative d(objective)/d(nodal displacement).  Unset => stock.
!     --------------------------------------------------------------------
      cnoarea=' '
      call getenv('CCX_NO_AREA_NORM',cnoarea)
!     
      do idesvar=1,ndesi
""")
    c = "            if(weightformgrad(idesvar).gt.0.d0) then\n"
    assert src.count(c) == 1
    src = src.replace(c, "            if((weightformgrad(idesvar).gt.0.d0).and.\n"
                         "     &         (cnoarea(1:1).ne.'1')) then\n")
    open(p, "w").write(src)
    print("   patched distributesens.f")

p = S + "/getdesiinfo3d.f"
src = open(p).read()
if "CCX_DESI_INTERIOR" in src:
    print("   getdesiinfo3d.f already patched")
else:
    d = "     &  ifacet(6,4),ifacew1(4,5),ifacew2(8,5),nopem,nk\n"
    assert src.count(d) == 1
    src = src.replace(d, d + "!\n      character*132 cintr\n")
    loop = """      do i=1,nk
         if(nodedesiinv(i).eq.-1) then
!
            write(*,*) '*WARNING in getdesiinfo:'
"""
    assert src.count(loop) == 1
    src = src.replace(loop, """!
!     ---- CCX_DESI_INTERIOR patch ---------------------------------------
!     A node still flagged -1 here is one that does not lie on an external
!     face with enough design corners -- i.e. an interior node.  Stock ccx
!     deletes it, because a SHAPE sensitivity is undefined without a surface
!     normal.  Under CCX_DESI_AXIS there is no normal, the perturbation is a
!     plain coordinate move, and the downstream machinery (elemperdesi,
!     createnodeneigh, createelemneigh) is node->element based and never
!     looks at faces -- so the node can simply be kept.  Unset => stock.
!     ---------------------------------------------------------------------
      cintr=' '
      call getenv('CCX_DESI_INTERIOR',cintr)
!
      do i=1,nk
         if(nodedesiinv(i).eq.-1) then
            if(cintr(1:1).eq.'1') then
               nodedesiinv(i)=1
               cycle
            endif
!
            write(*,*) '*WARNING in getdesiinfo:'
""")
    open(p, "w").write(src)
    print("   patched getdesiinfo3d.f")
PYEOF

echo "==> building (this compiles ~1200 sources)"
cat > "$SRC/Makefile" <<EOF
CFLAGS = -Wall -O2 -I $BUILD/toolroot/usr/include/spooles -DARCH="Linux" -DSPOOLES -DARPACK -DMATRIXSTORAGE -DNETWORKOUT
FFLAGS = -O2 -fallow-argument-mismatch
CC=gcc
FC=$BUILD/toolroot/usr/bin/gfortran-11
.c.o :
	\$(CC) \$(CFLAGS) -c \$<
.f.o :
	\$(FC) \$(FFLAGS) -c \$<
include Makefile.inc
SCCXMAIN = ccx_2.17.c
OCCXF = \$(SCCXF:.f=.o)
OCCXC = \$(SCCXC:.c=.o)
OCCXMAIN = \$(SCCXMAIN:.c=.o)
LIBS = $BUILD/toolroot/usr/lib/x86_64-linux-gnu/libspooles.a \\
       -L$BUILD/linkdir -larpack -llapack -lblas -latlas -lgfortran \\
       -lpthread -lm -lc
ccx_2.17: \$(OCCXMAIN) ccx_2.17.a
	\$(FC) -O2 -o \$@ \$(OCCXMAIN) ccx_2.17.a \$(LIBS)
ccx_2.17.a: \$(OCCXF) \$(OCCXC)
	ar r \$@ \$?
EOF
cd "$SRC"
./date.pl
make -j"$(nproc)" ccx_2.17

install -m 755 ccx_2.17 "$OUT"
echo
echo "==> installed $OUT"
echo "    run it with LD_LIBRARY_PATH=$CCXLIB"
echo "    with no CCX_* variables set it behaves exactly like stock ccx."
