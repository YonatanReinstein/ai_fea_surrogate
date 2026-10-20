/*****************************************************************************
*   Constructs locally varying trivariate tiles in microstructure            *
* constructions using a planar 4-leg cross tile, full depth along W.         *
*                                                                            *
*   Each tile is IRIT's own 3D cross tile (IritUserMicro3DCrossTile) with    *
* square legs, reduced to its four in-plane legs (+-U, +-V): the centre      *
* block and those four legs are cut at mid depth (W = 0.5) and the cuts are  *
* extruded through the full tile depth W in [0, 1].  The two +-W legs are    *
* dropped.  The per-tile parameter (dims.json d1..dK) is the leg half-width, *
* IRIT's OuterRadius: the centre square is 2t wide, and a leg's end on a    *
* tile face uses the mean of the two tiles sharing that face (the tile's     *
* own value on the part's outer boundary; CROSS4_LEG_RULE=min for the       *
* minimum), so neighbouring tiles meet with identical leg ends and the       *
* mesh stays conforming.  Exception: the loaded tile's end (free end, row   *
* FEA_LOAD_ROW, middle by default) has a fixed half-width,                   *
* CROSS4_FIXED_END_RADIUS.                                                   *
******************************************************************************
* (C) Gershon Elber, Technion, Israel Institute of Technology                *
******************************************************************************
* Based on the hollow-cube model (data/hollow_cube/CAD_model/model_lin.c).   *
*****************************************************************************/

#include <string.h>
#include "inc_irit/irit_sm.h"
#include "inc_irit/iritprsr.h"
#include "inc_irit/allocate.h"
#include "inc_irit/attribut.h"
#include "inc_irit/geom_lib.h"
#include "inc_irit/cagd_lib.h"
#include "inc_irit/triv_lib.h"
#include "inc_irit/mvar_lib.h"
#include "inc_irit/user_lib.h"
#include "inc_irit/grap_lib.h"

/* Keep every leg half-width below 0.5 so the legs and the centre stay strictly */
/* inside the unit cell (at 0.5 the cross fills the cell and the gaps between  */
/* legs vanish to zero-width slivers).                                          */
#define CROSS4_MAX_RADIUS 0.48
#define CROSS4_MAX_CENTRE 0.42      /* Max half-width of the (widened) centre; CROSS4_MAX_CENTRE env overrides. 0.44-0.46 leave legs too short for BndryShape's loft: whole-centre inversions near the cap (tested 2026-10-08). */
#define CROSS4_FIXED_END_RADIUS 0.0  /* Loaded tile's leg end: 0 (default since 2026-10-14) = follows the leg rule like every tile; > 0 fixes its half-width (0.25 before). CROSS4_FIXED_END_RADIUS env overrides. */
#define CROSS4_PAD_THICKNESS 0.1     /* Solid pad on the anchored face; CROSS4_PAD_THICKNESS env overrides, 0 = none. */
#define CROSS4_LOAD_PAD_THICKNESS 0.1  /* Solid pad on the loaded tile's face; CROSS4_LOAD_PAD_THICKNESS env overrides, 0 = none. */
#define CROSS4_PAD_FILLET 0.1        /* Leg-to-pad fillet radius (both pads); CROSS4_PAD_FILLET env overrides, 0 = sharp. */
#define CROSS4_LOAD_FLARE_HALFWIDTH 0.45 /* Load pad: the leg flares to this fixed half-width, so the loaded face never moves. */
#define CROSS4_CNTR_SHAPE 0.2       /* Junction rounding (IRIT CntrShape); valid with CROSS4_MAX_CENTRE <= 0.42. */
#define CROSS4_FLARE_CURV 2.0       /* Per-leg CntrShape rule (FlareLimitedCntr): flare tightness limit; CROSS4_FLARE_CURV env overrides, 0 = same CntrShape on every leg. */
#define CROSS4_BNDRY_SHAPE 0.2       /* Leg loft (IRIT BndryShape); with CntrShape 0.2 needs CROSS4_MAX_CENTRE <= 0.42. */

typedef struct Cross4LocalDataStruct {
    int NX, NY, NZ;
    CagdRType *ShellThicknesses;  /* NX*NY*NZ values, X-outermost order */
} Cross4LocalDataStruct;

static int read_fixed_dims(int *NX, int *NY, int *NZ);
static int read_shell_thicknesses(int NX, int NY, int NZ,
                                  CagdRType **ShellThicknesses);
static IritPrsrObjectStruct *BuildCross4Tile(const CagdRType FaceR[4],
                                             CagdRType CentreR,
                                             const CagdRType Pad[2],
                                             const CagdRType Squeeze[2],
                                             const int Extend[2],
                                             CagdRType Fillet,
                                             const int Refine[2]);
static CagdRType SumTrivarVolumes(IritPrsrObjectStruct *PObj);
static IritPrsrObjectStruct *PreProcessTile(IritPrsrObjectStruct *Tile,
    IritUserMicroPreProcessTileCBStruct *CBData);
static void GenerateMicroStructures(void);

/*****************************************************************************
* DESCRIPTION:
*   Read tiling grid dimensions (NX, NY, NZ) from fixed_dims.itd.
*   List format: [NX, NY, NZ]
*
* RETURN VALUE:
*   int: TRUE on success, FALSE on failure.
*****************************************************************************/
static int read_fixed_dims(int *NX, int *NY, int *NZ)
{
    IritPrsrObjectStruct *PObj;
    const char *FileName = "fixed_dims.itd";

    PObj = IritPrsrGetDataFiles(&FileName, 1, FALSE, FALSE);
    if (PObj == NULL) {
        fprintf(stderr, "Failed to load fixed_dims.itd\n");
        return FALSE;
    }

    if (!IRIT_PRSR_IS_NUM_OBJ(PObj)) { fprintf(stderr, "NX missing in fixed_dims.itd\n"); return FALSE; }
    *NX = (int)PObj->U.R;
    PObj = PObj->Pnext;

    if (PObj == NULL || !IRIT_PRSR_IS_NUM_OBJ(PObj)) { fprintf(stderr, "NY missing in fixed_dims.itd\n"); return FALSE; }
    *NY = (int)PObj->U.R;
    PObj = PObj->Pnext;

    if (PObj == NULL || !IRIT_PRSR_IS_NUM_OBJ(PObj)) { fprintf(stderr, "NZ missing in fixed_dims.itd\n"); return FALSE; }
    *NZ = (int)PObj->U.R;

    return TRUE;
}

/*****************************************************************************
* DESCRIPTION:
*   Read per-tile ShellThickness values from dims.itd.
*   List format: [st_0, st_1, ..., st_{NX*NY*NZ-1}]
*   Values are stored in X-outermost (U-outermost) order.
*
* RETURN VALUE:
*   int: TRUE on success, FALSE on failure.
*****************************************************************************/
static int read_shell_thicknesses(int NX, int NY, int NZ,
                                  CagdRType **ShellThicknesses)
{
    IritPrsrObjectStruct *PObj;
    const char *DimsFileName = "dims.itd";
    int i, n;

    PObj = IritPrsrGetDataFiles(&DimsFileName, 1, FALSE, FALSE);
    if (PObj == NULL) {
        fprintf(stderr, "Failed to load dims.itd\n");
        return FALSE;
    }

    n = NX * NY * NZ;
    *ShellThicknesses = (CagdRType *)IritMalloc(sizeof(CagdRType) * n);

    for (i = 0; i < n; i++) {
        if (PObj == NULL || !IRIT_PRSR_IS_NUM_OBJ(PObj)) {
            fprintf(stderr, "ShellThickness[%d] missing\n", i);
            IritFree(*ShellThicknesses);
            return FALSE;
        }
        (*ShellThicknesses)[i] = PObj->U.R;
        PObj = PObj->Pnext;
    }

    return TRUE;
}

/*****************************************************************************
* DESCRIPTION:
*   Cut TV at mid depth (W = 0.5 in space) and extrude the cut through the
*   full tile depth.  The TV's parametric direction that runs along W is the
*   one whose mid iso-surface is flattest in W; that surface is projected to
*   W = 0 and extruded by (0, 0, 1).
*****************************************************************************/
static IritTrivTVStruct *ExtrudeMidDepth(const IritTrivTVStruct *TV)
{
    int d, i, Best = -1;
    CagdRType BestExt = IRIT_INFNTY;
    IritTrivTVDirType Dirs[3] = { IRIT_TRIV_CONST_U_DIR, IRIT_TRIV_CONST_V_DIR,
                                  IRIT_TRIV_CONST_W_DIR };
    IritCagdSrfStruct *Srf;
    IritCagdVecStruct Vec;
    IritTrivTVStruct *ETV;

    for (d = 0; d < 3; d++) {
        CagdRType UMin, UMax, VMin, VMax, WMin, WMax, t;
        IritGeomBBBboxStruct BB;

        IritTrivTVDomain(TV, &UMin, &UMax, &VMin, &VMax, &WMin, &WMax);
        t = d == 0 ? (UMin + UMax) * 0.5 : d == 1 ? (VMin + VMax) * 0.5
                                                    : (WMin + WMax) * 0.5;
        Srf = IritTrivSrfFromTV(TV, t, Dirs[d], FALSE);
        IritCagdSrfBBox(Srf, &BB);
        if (BB.Max[2] - BB.Min[2] < BestExt) {
            BestExt = BB.Max[2] - BB.Min[2];
            Best = d;
        }
        IritCagdSrfFree(Srf);
    }
    if (Best < 0 || BestExt > 1e-6) {
        fprintf(stderr, "ExtrudeMidDepth: no planar mid-depth cut (%g)\n", BestExt);
        return NULL;
    }

    {
        CagdRType UMin, UMax, VMin, VMax, WMin, WMax, t;

        IritTrivTVDomain(TV, &UMin, &UMax, &VMin, &VMax, &WMin, &WMax);
        t = Best == 0 ? (UMin + UMax) * 0.5 : Best == 1 ? (VMin + VMax) * 0.5
                                                         : (WMin + WMax) * 0.5;
        Srf = IritTrivSrfFromTV(TV, t, Dirs[Best], FALSE);
    }

    /* Project the (planar, mid-depth) cut onto W = 0. */
    {
        int n = Srf -> ULength * Srf -> VLength,
            Rational = IRIT_CAGD_IS_RATIONAL_SRF(Srf);

        for (i = 0; i < n; i++)
            Srf -> Points[3][i] = 0.0;
        if (Rational)
            fprintf(stderr, "ExtrudeMidDepth: unexpected rational cut\n");
    }

    IRIT_ZAP_MEM(&Vec, sizeof(IritCagdVecStruct));
    Vec.Vec[2] = 1.0;
    ETV = IritTrivExtrudeTV(Srf, &Vec);
    IritCagdSrfFree(Srf);

    if (getenv("CROSS4_DEBUG") != NULL)
        fprintf(stderr, "piece: source TV neg-Jacobian %d, extruded neg-Jacobian %d\n",
                IritMvarTVRglrIsNegJacobian(TV), IritMvarTVRglrIsNegJacobian(ETV));

    return ETV;
}

/*****************************************************************************
* DESCRIPTION:
*   A full-depth box U in [X0, X1], V in [Y0, Y1], W in [0, 1].
*****************************************************************************/
static IritTrivTVStruct *PlanarBox(CagdRType X0, CagdRType Y0,
                                   CagdRType X1, CagdRType Y1)
{
    IritCagdVecStruct Vec;
    IritCagdSrfStruct
        *Srf = IritCagdPrimPlaneSrf(X0, Y0, X1, Y1, 0.0);
    IritTrivTVStruct *TV;

    IRIT_ZAP_MEM(&Vec, sizeof(IritCagdVecStruct));
    Vec.Vec[2] = 1.0;
    TV = IritTrivExtrudeTV(Srf, &Vec);
    IritCagdSrfFree(Srf);
    return TV;
}

/*****************************************************************************
* DESCRIPTION:
*   Leg-to-pad fillet block, full depth: bounded by two quarter ellipses
*   with semi-axes Rho (along U) and Ry (along V) -- quarter circles when
*   Ry = Rho -- as cubic Beziers (k = 0.5523), tangent to the leg's sides
*   (V = 0.5 -+ r) and to the pad's face (U = XFace).  It spans U from XFace
*   (width 2 (r + Ry)) to XFace + Dir * Rho (width 2 r, the leg's end).  Dir
*   is +1 for a U-min pad, -1 for a U-max pad.
*****************************************************************************/
static IritTrivTVStruct *FilletTV(CagdRType XFace, CagdRType Dir,
                                  CagdRType r, CagdRType Rho, CagdRType Ry)
{
    const CagdRType k = 0.5523;
    int i, s;
    IritCagdCrvStruct *Arc[2];
    IritCagdSrfStruct *Srf;
    IritCagdVecStruct Vec;
    IritTrivTVStruct *TV;

    for (s = 0; s < 2; s++) {
        CagdRType Sgn = s == 0 ? 1.0 : -1.0,              /* Low / high side. */
            Cy = 0.5 - Sgn * (r + Ry),            /* Pad-face end of the arc. */
            X[4], Y[4];

        X[0] = XFace;                         Y[0] = Cy;
        X[1] = XFace;                         Y[1] = Cy + Sgn * k * Ry;
        X[2] = XFace + Dir * (1.0 - k) * Rho; Y[2] = Cy + Sgn * Ry;
        X[3] = XFace + Dir * Rho;             Y[3] = Cy + Sgn * Ry;
        Arc[s] = IritCagdBzrCrvNew(4, IRIT_CAGD_PT_E3_TYPE);
        for (i = 0; i < 4; i++) {
            Arc[s] -> Points[1][i] = X[i];
            Arc[s] -> Points[2][i] = Y[i];
            Arc[s] -> Points[3][i] = 0.0;
        }
        if (Dir < 0.0) {        /* Keep the (u, v, w) frame right-handed. */
            IritCagdCrvStruct
                *R = IritCagdCrvReverse(Arc[s]);

            IritCagdCrvFree(Arc[s]);
            Arc[s] = R;
        }
    }
    Srf = IritCagdRuledSrf(Arc[0], Arc[1], 2, 2);
    IritCagdCrvFree(Arc[0]);
    IritCagdCrvFree(Arc[1]);
    IRIT_ZAP_MEM(&Vec, sizeof(IritCagdVecStruct));
    Vec.Vec[2] = 1.0;
    TV = IritTrivExtrudeTV(Srf, &Vec);
    IritCagdSrfFree(Srf);
    return TV;
}

/*****************************************************************************
* DESCRIPTION:
*   Local mesh refinement without touching irit2inp's single global U,V,W
*   resolution: every piece of the tile is split into K equal parametric
*   parts along whichever of its (u, v) directions runs along tile axis Axis
*   (0 = the tile's U/x, along the grid's columns; 1 = its V/y, along rows).
*   irit2inp then meshes each part with the global resolution, i.e. that
*   direction gets K times the elements.  Pieces sharing a face are split
*   identically when the whole row/column is split, so faces stay conforming.
*****************************************************************************/
static IritTrivTVStruct *SplitAlongAxis(IritTrivTVStruct *List, int Axis, int K)
{
    IritTrivTVStruct *TV, *Next, *Out = NULL;

    if (K <= 1)
        return List;
    for (TV = List; TV != NULL; TV = Next) {
        int ULen = TV -> ULength, VLen = TV -> VLength, j;
        CagdRType **P = TV -> Points, du, dv,
            UMin, UMax, VMin, VMax, WMin, WMax, t0, t1;
        IritTrivTVDirType Dir;
        IritTrivTVStruct *Rest;

        Next = TV -> Pnext;
        TV -> Pnext = NULL;
        /* Alignment of the u and v edges (control-mesh corners) with Axis: */
        /* the share of each edge's in-plane length along Axis.  Raw spans  */
        /* are not enough -- a thin tapered leg's taper along its length    */
        /* can exceed its width.                                           */
        {
            int Ou = ULen - 1, Ov = (VLen - 1) * ULen, Oth = 1 - Axis;
            CagdRType au = fabs(P[Axis + 1][Ou] - P[Axis + 1][0]),
                bu = fabs(P[Oth + 1][Ou] - P[Oth + 1][0]),
                av = fabs(P[Axis + 1][Ov] - P[Axis + 1][0]),
                bv = fabs(P[Oth + 1][Ov] - P[Oth + 1][0]);

            du = au / (au + bu + 1e-30);
            dv = av / (av + bv + 1e-30);
        }
        Dir = du >= dv ? IRIT_TRIV_CONST_U_DIR : IRIT_TRIV_CONST_V_DIR;
        IritTrivTVDomain(TV, &UMin, &UMax, &VMin, &VMax, &WMin, &WMax);
        t0 = Dir == IRIT_TRIV_CONST_U_DIR ? UMin : VMin;
        t1 = Dir == IRIT_TRIV_CONST_U_DIR ? UMax : VMax;
        Rest = TV;
        for (j = 1; j < K; j++) {
            /* Split the remainder at 1/(K-j+1) of ITS OWN current domain: a  */
            /* subdivided piece may be re-parametrized, so the original     */
            /* domain's fractions do not carry over.                        */
            IritTrivTVStruct *Two, *Second;

            IritTrivTVDomain(Rest, &UMin, &UMax, &VMin, &VMax, &WMin, &WMax);
            t0 = Dir == IRIT_TRIV_CONST_U_DIR ? UMin : VMin;
            t1 = Dir == IRIT_TRIV_CONST_U_DIR ? UMax : VMax;
            Two = IritTrivTVSubdivAtParam(Rest, t0 + (t1 - t0) / (K - j + 1), Dir);
            Second = Two -> Pnext;

            Two -> Pnext = Out;
            Out = Two;
            if (Rest != TV)
                IritTrivTVFree(Rest);
            Rest = Second;
        }
        if (Rest != TV)
            IritTrivTVFree(TV);
        Rest -> Pnext = Out;
        Out = Rest;
    }
    return Out;
}

/*****************************************************************************
* DESCRIPTION:
*   Per-leg CntrShape from a flare-tightness limit (CROSS4_FLARE_CURV > 0).
*   IRIT's solid branch (micro1tile.c, C^1 joints, BndryShape > 0) builds a
*   leg through four sections, in IRIT's [-1, 1] frame along the leg axis:
*     centre face   at depth Rc,          half-width Rc
*     Srf1          at depth Rc (1 + c),  half-width Rc (1 - c)
*     end copy      at depth 2/3 + D1/3,  half-width 2r   (D1 = Srf1 depth)
*     end (face)    at depth 1,           half-width 2r
*   so it reaches the face straight (tangent to the neighbour's leg).  The
*   width step Rc - 2r happens over the stretch between Srf1 and the face, so
*   the bend's tightness (a curvature proxy) is
*     K(c) = |Rc - 2r| / (1 - Rc (1 + c))^2,
*   about 1.6 for c = 0.1 on uniform t = 0.3 (converged) and 44 for c = 0.2
*   (a neck that converges very slowly).
*   Rc is the centre half-width, 2 R (1 + 1.5 d) when all radii are equal
*   and 2 CentreR (1 + d) otherwise, d = mean CntrShape of the six faces.
*   Each in-plane leg gets the largest c in [CMin, CMax] with K(c) <= SMax
*   (closed form; CMin if none qualifies); the dropped +-W legs get
*   the in-plane mean.  Two passes settle the d <-> Rc coupling.
*****************************************************************************/
static void FlareLimitedCntr(const CagdRType FaceR[4], CagdRType CentreR,
                             CagdRType CMax, CagdRType CMin, CagdRType SMax,
                             CagdRType MaxCentre, CagdRType Cntr[6])
{
    int i, pass, Equal = TRUE;

    /* BuildCross4Tile passes the +-W radius as 1.001 CentreR, so IRIT never */
    /* takes its equal-radii branch: centre half-width = CentreR (1 + d).   */
    Equal = FALSE;
    /* The centre size depends on the mean d of the legs' values, and each  */
    /* leg's best value falls as d grows (bigger centre -> tighter flare),  */
    /* so d = mean_i c_i(d) has one solution: bisect on it.                 */
    {
        CagdRType Lo = CMin, Hi = CMax, d = CMax;

        for (pass = 0; pass < 40; pass++) {
            CagdRType Rc = IRIT_MIN(Equal ? 2.0 * CentreR * (1.0 + 1.5 * d)
                                          : 2.0 * CentreR * 1.001 * (1.0 + d),
                                    2.0 * MaxCentre),
                Sum = 0.0;

            for (i = 0; i < 4; i++) {
                /* K(c) = |Rc - 2r| / (1 - Rc (1 + c))^2 rises with c, so   */
                /* the largest c with K(c) <= SMax is closed form; clamped  */
                /* to [CMin, CMax] (CMin also when even CMin is too tight).  */
                /* Continuous in the radii, so the geometry -- and the      */
                /* design gradient -- has no steps.                          */
                CagdRType Step = fabs(Rc - 2.0 * FaceR[i]),
                    c = (1.0 - sqrt(Step / SMax)) / Rc - 1.0;

                Cntr[i] = IRIT_BOUND(c, CMin, CMax);
                Sum += Cntr[i];
            }
            if (pass == 39)
                break;
            if (Sum / 4.0 > d)
                Lo = d;
            else
                Hi = d;
            d = 0.5 * (Lo + Hi);
            if (pass == 0)
                d = 0.5 * (CMin + CMax);
        }
        Cntr[4] = Cntr[5] = (Cntr[0] + Cntr[1] + Cntr[2] + Cntr[3]) / 4.0;
    }
}

/*****************************************************************************
* DESCRIPTION:
*   Build a planar 4-leg cross tile in the unit cube [0, 1]^3.
*
* PARAMETERS:
*   FaceR:   Leg half-widths (IRIT OuterRadius) at the four in-plane faces, in
*            order U-min, U-max, V-min, V-max.  Each in (0, 0.5).
*   CentreR: Half-width of the centre square (the tile's own parameter).
*            Passed as the radius of the two dropped +-W legs: IRIT sizes the
*            centre along x from max(Ry, Rz) and along y from max(Rx, Rz), so
*            with Rz = CentreR >= every FaceR the centre is a 2 CentreR square
*            and each leg tapers from it to its FaceR end.
*   Pad:     Solid pad thickness on the U-min / U-max face (0 = none).
*   Squeeze: The cross is squeezed into U in [Squeeze[0], 1 - Squeeze[1]].
*            Constant over a whole column of tiles (pad + fillet), so the
*            faces between tiles of that column stay identical.
*   Extend:  TRUE on a side that is squeezed but has no pad: a straight box
*            extends the leg back to the face (the free-end tiles beside the
*            loaded one).
*   Fillet:  Leg-to-pad fillet radius (0 = a sharp 90-degree joint).
*
* RETURN VALUE:
*   IritPrsrObjectStruct *: A TRIVAR object of the centre, the four legs and
*                           any pad, fillet and extension pieces, each
*                           spanning W in [0, 1], or NULL on error.
*****************************************************************************/
static IritPrsrObjectStruct *BuildCross4Tile(const CagdRType FaceR[4],
                                             CagdRType CentreR,
                                             const CagdRType Pad[2],
                                             const CagdRType Squeeze[2],
                                             const int Extend[2],
                                             CagdRType Fillet,
                                             const int Refine[2])
{
    int i, s;
    char *Err = NULL;
    IritUserMicroTileBndryPrmStruct P[6];
    IritPrsrObjectStruct *Cross;
    IritTrivTVStruct *TV, *Kept = NULL;
    CagdRType LegCntr[6];

    /* Rounded junctions: IRIT's CntrShape widens the centre block and        */
    /* BndryShape makes each leg a cubic loft that flares from that wider     */
    /* centre into the leg, instead of a sharp 90-degree corner.  The centre's */
    /* half-width grows to about RMax * (1 + 1.5 * CntrShape), so CntrShape is */
    /* capped per tile to keep the centre inside the cell: thick tiles get     */
    /* less rounding.  CROSS4_CNTR_SHAPE / CROSS4_BNDRY_SHAPE override the     */
    /* defaults (0 = the sharp cross).                                         */
    CagdRType RMax = IRIT_MAX(IRIT_MAX(IRIT_MAX(FaceR[0], FaceR[1]),
                                       IRIT_MAX(FaceR[2], FaceR[3])), CentreR),
        Cntr = getenv("CROSS4_CNTR_SHAPE") != NULL ?
                   atof(getenv("CROSS4_CNTR_SHAPE")) : CROSS4_CNTR_SHAPE,
        Bndry = getenv("CROSS4_BNDRY_SHAPE") != NULL ?
                   atof(getenv("CROSS4_BNDRY_SHAPE")) : CROSS4_BNDRY_SHAPE,
        MaxCentre = getenv("CROSS4_MAX_CENTRE") != NULL ?
                   atof(getenv("CROSS4_MAX_CENTRE")) : CROSS4_MAX_CENTRE,
        CntrMax = (MaxCentre / RMax - 1.0) / 1.5;

    Cntr = IRIT_BOUND(Cntr, 0.0, IRIT_MAX(CntrMax, 0.0));

    /* Optional per-leg rounding (CROSS4_FLARE_CURV > 0): each leg's        */
    /* CntrShape is the largest value up to the (capped) Cntr above whose   */
    /* flare stays below the tightness limit, never below CROSS4_CNTR_MIN, see */
    /* FlareLimitedCntr.  Otherwise every leg uses Cntr.                     */
    {
        const char *SlStr = getenv("CROSS4_FLARE_CURV"),
            *MinStr = getenv("CROSS4_CNTR_MIN");
        CagdRType SMax = SlStr != NULL ? atof(SlStr) : CROSS4_FLARE_CURV,
            CMin = MinStr != NULL ? atof(MinStr) : 0.05;

        if (SMax > 0.0)
            FlareLimitedCntr(FaceR, CentreR, Cntr, IRIT_MIN(CMin, Cntr), SMax,
                             MaxCentre, LegCntr);
        else
            for (i = 0; i < 6; i++)
                LegCntr[i] = Cntr;
        if (getenv("CROSS4_DEBUG") != NULL)
            fprintf(stderr, "cntr: FaceR %.3f %.3f %.3f %.3f CentreR %.3f -> %.4f %.4f %.4f %.4f\n",
                    FaceR[0], FaceR[1], FaceR[2], FaceR[3], CentreR,
                    LegCntr[0], LegCntr[1], LegCntr[2], LegCntr[3]);
    }

    for (i = 0; i < 6; i++) {
        IRIT_ZAP_MEM(&P[i], sizeof(IritUserMicroTileBndryPrmStruct));
        P[i].Circular = FALSE;                           /* Square legs. */
        P[i].CntrShape = LegCntr[i];
        P[i].BndryShape = Bndry;
        /* The +-W legs are dropped; their radius sets the centre size.  */
        /* 0.1% above CentreR so the radii are never all "equal": IRIT  */
        /* sizes the centre R (1 + 1.5 d) when they are (within ~1e-5)  */
        /* and R (1 + d) otherwise, so a tile whose neighbours were all */
        /* at least as thick jumped (5% centre, 1.3% volume, 5% stress) */
        /* the moment one got thinner.  Always the second formula now.  */
        P[i].OuterRadius = i < 4 ? FaceR[i] : CentreR * 1.001;
    }

    Cross = IritUserMicro3DCrossTile(&P[0], &P[1], &P[2], &P[3], &P[4], &P[5],
                                     FALSE, &Err);
    if (Cross == NULL || !IRIT_PRSR_IS_TRIVAR_OBJ(Cross)) {
        fprintf(stderr, "IritUserMicro3DCrossTile failed: %s\n",
                Err != NULL ? Err : "?");
        return NULL;
    }

    /* Keep the centre and the four in-plane legs: everything that does not */
    /* reach W = 0 or W = 1 (the two +-W legs do).                           */
    for (TV = Cross -> U.Trivars; TV != NULL; TV = TV -> Pnext) {
        IritGeomBBBboxStruct BB;
        IritTrivTVStruct *ETV;

        IritTrivTVBBox(TV, &BB);
        if (BB.Min[2] < 1e-9 || BB.Max[2] > 1.0 - 1e-9)
            continue;
        if ((ETV = ExtrudeMidDepth(TV)) == NULL) {
            IritTrivTVFreeList(Kept);
            IritPrsrFreeObject(Cross);
            return NULL;
        }
        ETV -> Pnext = Kept;
        Kept = ETV;
    }
    IritPrsrFreeObject(Cross);

    {
        int n = 0;

        for (TV = Kept; TV != NULL; TV = TV -> Pnext)
            n++;
        if (n != 5) {
            fprintf(stderr, "BuildCross4Tile: expected 5 pieces, got %d\n", n);
            IritTrivTVFreeList(Kept);
            return NULL;
        }
    }

    /* Pads (the role of IRIT's shell): the cross is squeezed in U to make  */
    /* room and each padded face gets a solid plate, full height and depth. */
    /* IRIT's own shell is one 1x1 patch per tile, which irit2inp meshes    */
    /* independently of the leg it touches (non-conforming); here the plate */
    /* is three boxes split where the fillet (or leg) meets it, V in        */
    /* [0, .5-r-Rho], [.5-r-Rho, .5+r+Rho], [.5+r+Rho, 1], so every shared  */
    /* face is identical on both sides.                                     */
    if (Squeeze[0] > 0.0 || Squeeze[1] > 0.0) {
        IrtHmgnMatType MScl, MTrn, M;

        IritMiscMatGenMatScale(1.0 - Squeeze[0] - Squeeze[1], 1.0, 1.0, MScl);
        IritMiscMatGenMatTrans(Squeeze[0], 0.0, 0.0, MTrn);
        IritMiscMatMultTwo4by4(M, MScl, MTrn);
        for (TV = Kept; TV != NULL; TV = TV -> Pnext)
            IritTrivTVMatTransform2(TV, M);
    }
    for (s = 0; s < 2; s++) {
        CagdRType r = FaceR[s],
            Dir = s == 0 ? 1.0 : -1.0,
            XFace = s == 0 ? Pad[0] : 1.0 - Pad[1],         /* Pad's inner face. */
            XLeg = s == 0 ? Squeeze[0] : 1.0 - Squeeze[1];  /* Squeezed leg end. */
        IritTrivTVStruct *New[4];
        int nNew = 0;

        if (Pad[s] > 0.0) {
            /* Anchored pad (s = 0): circular fillet, the pad split where it */
            /* lands (clamped nodes may slide along the face).  Load pad     */
            /* (s = 1): the leg flares to the FIXED half-width              */
            /* CROSS4_LOAD_FLARE_HALFWIDTH, so the pad's split points -- and */
            /* every node of the loaded (rigid-tied) face -- never move with */
            /* the design; the exact adjoint assumes a design-fixed tie.     */
            CagdRType Rho = Fillet,
                Ry = Rho <= 0.0 ? 0.0      /* No fillet: split at the leg. */
                   : s == 0 ? Fillet : CROSS4_LOAD_FLARE_HALFWIDTH - r,
                VCut[4] = { 0.0, 0.5 - r - Ry, 0.5 + r + Ry, 1.0 },
                X0 = s == 0 ? 0.0 : 1.0 - Pad[1],
                X1 = s == 0 ? Pad[0] : 1.0;

            for (i = 0; i < 3; i++)
                New[nNew++] = PlanarBox(X0, VCut[i], X1, VCut[i + 1]);
            if (Rho > 0.0)
                New[nNew++] = FilletTV(XFace, Dir, r, Rho, Ry);
        }
        else if (Extend[s] && Squeeze[s] > 0.0)
            New[nNew++] = s == 0 ? PlanarBox(0.0, 0.5 - r, XLeg, 0.5 + r)
                                 : PlanarBox(XLeg, 0.5 - r, 1.0, 0.5 + r);
        for (i = 0; i < nNew; i++) {
            New[i] -> Pnext = Kept;
            Kept = New[i];
        }
    }

    Kept = SplitAlongAxis(Kept, 0, Refine[0]);
    Kept = SplitAlongAxis(Kept, 1, Refine[1]);

    Cross = IritPrsrGenTRIVARObject(Kept);
    IritMiscAttrIDSetObjectRGBColor(Cross, 255, 0, 0);
    return IritPrsrGenLISTObject(Cross);
}

/*****************************************************************************
* DESCRIPTION:
*   Recursively sum |volume| of every trivariate reachable from PObj.  PObj
*   may be a trivariate object (with Pnext-linked TVs), a list object (whose
*   entries are walked recursively), or anything else (contributes 0).
*
*   This copes with the microstructure composition returning either a list of
*   per-tile objects (multi-tile case) or a bare trivariate (the single
*   1x1x1 tile case), avoiding the invalid list dereference that the latter
*   would otherwise trigger.
*
* RETURN VALUE:
*   CagdRType: The accumulated absolute trivariate volume.
*****************************************************************************/
static CagdRType SumTrivarVolumes(IritPrsrObjectStruct *PObj)
{
    CagdRType Vol = 0.0;

    if (PObj == NULL)
        return 0.0;

    if (IRIT_PRSR_IS_TRIVAR_OBJ(PObj)) {
        IritTrivTVStruct *TV;

        for (TV = PObj->U.Trivars; TV != NULL; TV = TV->Pnext)
            Vol += fabs(IritTrivTVVolume(TV, TRUE));
    }
    else if (IRIT_PRSR_IS_OLST_OBJ(PObj)) {
        IritPrsrObjectStruct *MQ;
        int j;

        for (j = 0; (MQ = IritPrsrListObjectGet(PObj, j)) != NULL; j++)
            Vol += SumTrivarVolumes(MQ);
    }

    return Vol;
}

/*****************************************************************************
* DESCRIPTION:
*   Callback invoked for each tile in the 3D grid.  Looks up the
*   ShellThickness for this tile by its (ix, iy, iz) index, then constructs
*   a hollow-cube tile and maps it through the deformation matrix.
*
* PARAMETERS:
*   Tile:   Expected NULL (we build the tile here from scratch).
*   CBData: Framework callback data including tile indices and transform.
*
* RETURN VALUE:
*   IritPrsrObjectStruct *: The constructed and transformed tile.
*****************************************************************************/
static IritPrsrObjectStruct *PreProcessTile(IritPrsrObjectStruct *Tile,
    IritUserMicroPreProcessTileCBStruct *CBData)
{
    Cross4LocalDataStruct
        *LclData = (Cross4LocalDataStruct *)CBData->CBFuncData;
    int ix = CBData->TileIdxs[0];
    int iy = CBData->TileIdxs[1];
    int iz = CBData->TileIdxs[2];
    int idx = ix * LclData->NY * LclData->NZ + iy * LclData->NZ + iz;
    CagdRType ShellThickness = LclData->ShellThicknesses[idx],
        FaceR[4];
    int k,
        Nbr[4][2] = { { ix - 1, iy }, { ix + 1, iy }, { ix, iy - 1 }, { ix, iy + 1 } };

    assert(Tile == NULL);

    /* Leg end on each in-plane face (the intersection with the neighbour):  */
    /* the MEAN of this tile's and the neighbour's parameter, so both tiles  */
    /* build the same end.  Outside the grid (the part's outer boundary) the */
    /* tile's own value.  Clamped below 0.5 so the legs and centre stay      */
    /* inside the unit cell.                                                 */
    /* CROSS4_LEG_RULE=min: the MINIMUM of the two parameters instead (the   */
    /* default before 2026-10-14; the missing outer neighbour counts as      */
    /* infinity, so the boundary is again the tile's own).                   */
    {
        const char *Rule = getenv("CROSS4_LEG_RULE");
        int Avg = Rule == NULL || Rule[0] == 0 || strcmp(Rule, "avg") == 0;

        if (!Avg && strcmp(Rule, "min") != 0) {
            fprintf(stderr, "CROSS4_LEG_RULE=%s: use avg or min\n", Rule);
            exit(1);
        }

        for (k = 0; k < 4; k++) {
            int jx = Nbr[k][0],
                jy = Nbr[k][1];
            CagdRType Other = IRIT_INFNTY;

            if (jx >= 0 && jx < LclData->NX && jy >= 0 && jy < LclData->NY)
                Other = LclData->ShellThicknesses[jx * LclData->NY * LclData->NZ +
                                                  jy * LclData->NZ + iz];
            FaceR[k] = Avg && Other < IRIT_INFNTY ? 0.5 * (ShellThickness + Other)
                                                   : IRIT_MIN(ShellThickness, Other);
            FaceR[k] = IRIT_MIN(FaceR[k], CROSS4_MAX_RADIUS);
        }
    }

    /* Pads, read from the environment (see the #defines):               */
    /*   anchored pad  every tile of the first column (U-min, z = 0)        */
    /*   load pad      only the loaded tile (U-max of the last column, in   */
    /*                 the load row FEA_LOAD_ROW -- the same variable       */
    /*                 boundary_conditions.load_row() reads, middle row by  */
    /*                 default).  Its outer face is the full 1x1 tile face, */
    /*                 fixed whatever the design, so the load stays         */
    /*                 design-independent.  Without a load pad the loaded   */
    /*                 leg end follows the leg rule (CROSS4_FIXED_END_RADIUS */
    /*                 > 0 fixes it instead).                             */
    /* A padded column is squeezed by the constant pad + fillet, and the    */
    /* other free-end tiles get a straight extension so their legs still    */
    /* reach the free end.  The leg end under a filleted pad is capped at   */
    /* 0.49 - Fillet so the fillet fits inside the cell.                    */
    {
        const char *LoadRowStr = getenv("FEA_LOAD_ROW"),
            *PadStr = getenv("CROSS4_PAD_THICKNESS"),
            *LPadStr = getenv("CROSS4_LOAD_PAD_THICKNESS"),
            *FilStr = getenv("CROSS4_PAD_FILLET");
        /* FEA_LOAD_ROW may list several rows ("4,5,6"): each gets a load    */
        /* pad, and adjacent pads meet in one plate (each spans its full     */
        /* tile height, split identically at the shared face).              */
        int InLoadRows = FALSE,
            IsLoad,
            Extend[2];

        if (LoadRowStr != NULL && LoadRowStr[0] != 0) {
            const char *c = LoadRowStr;

            while (*c != 0) {
                if (atoi(c) == iy)
                    InLoadRows = TRUE;
                while (*c != 0 && *c != ',')
                    c++;
                if (*c == ',')
                    c++;
            }
        }
        else
            InLoadRows = iy == LclData->NY / 2;
        IsLoad = ix == LclData->NX - 1 && InLoadRows;
        CagdRType PadT = PadStr != NULL && PadStr[0] != 0 ? atof(PadStr)
                                                          : CROSS4_PAD_THICKNESS,
            LPadT = LPadStr != NULL && LPadStr[0] != 0 ? atof(LPadStr)
                                                       : CROSS4_LOAD_PAD_THICKNESS,
            Fil = FilStr != NULL && FilStr[0] != 0 ? atof(FilStr)
                                                   : CROSS4_PAD_FILLET,
            Pad[2], Squeeze[2];

        Fil = IRIT_MAX(Fil, 0.0);
        Pad[0] = ix == 0 ? PadT : 0.0;
        Pad[1] = IsLoad ? LPadT : 0.0;
        Squeeze[0] = ix == 0 && PadT > 0.0 ? PadT + Fil : 0.0;
        Squeeze[1] = ix == LclData->NX - 1 && LPadT > 0.0 ? LPadT + Fil : 0.0;
        Extend[0] = FALSE;
        Extend[1] = ix == LclData->NX - 1 && !IsLoad;

        /* CROSS4_FIXED_END_RADIUS env overrides the fixed end; <= 0 leaves */
        /* the loaded tile's end to the leg rule, so the geometry no longer */
        /* depends on FEA_LOAD_ROW (multi-load datasets, one mesh per design). */
        {
            const char *FixStr = getenv("CROSS4_FIXED_END_RADIUS");
            CagdRType FixR = FixStr != NULL && FixStr[0] != 0 ? atof(FixStr)
                                                             : CROSS4_FIXED_END_RADIUS;

            if (IsLoad && LPadT <= 0.0 && FixR > 0.0)
                FaceR[1] = FixR;
        }
        for (k = 0; k < 2; k++)
            if (Pad[k] > 0.0 && Fil > 0.0)
                FaceR[k] = IRIT_MIN(FaceR[k], 0.49 - Fil);
        if (Pad[1] > 0.0 && Fil > 0.0)  /* Load flare needs room to widen. */
            FaceR[1] = IRIT_MIN(FaceR[1], CROSS4_LOAD_FLARE_HALFWIDTH - 0.02);

        /* The centre square is sized by this tile's own parameter (side   */
        /* 2t): with min() at every leg end, a tile thicker than its        */
        /* neighbours would otherwise build exactly like its thinnest one.  */
        /* Local refinement (CROSS4_REFINE_COL_K / CROSS4_REFINE_ROW_K,      */
        /* default 1): the loaded tile's column is split K times along x    */
        /* (the grid length) and its row K times along y (the height), see  */
        /* SplitAlongAxis.                                                  */
        {
            const char *KC = getenv("CROSS4_REFINE_COL_K"),
                *KR = getenv("CROSS4_REFINE_ROW_K");
            int Refine[2];

            Refine[0] = ix == LclData->NX - 1 && KC != NULL ? atoi(KC) : 1;
            Refine[1] = InLoadRows && KR != NULL ? atoi(KR) : 1;
            Tile = BuildCross4Tile(FaceR, IRIT_MIN(ShellThickness, CROSS4_MAX_RADIUS),
                                   Pad, Squeeze, Extend, Fil, Refine);
        }
    }

    if (Tile == NULL) {
        fprintf(stderr, "BuildCross4Tile failed at [%d,%d,%d] (t=%g)\n",
                ix, iy, iz, ShellThickness);
        return NULL;
    }

    Tile = IritGeomTransformObjectInPlace(Tile, CBData->Mat);

    return Tile;
}

/*****************************************************************************
* DESCRIPTION:
*   Create a micro structure tiled with planar 4-leg cross tiles.
*
* PARAMETERS:
*   None
*
* RETURN VALUE:
*   void
*****************************************************************************/
static void GenerateMicroStructures(void)
{
    const char *InputDefMap = "outline.itd";
    int i, Handler;
    IritPrsrObjectStruct *MS, *DefMapPObj;
    IritMvarMVStruct *DeformMV;
    IritTrivTVStruct *TVMap;
    IritUserMicroParamStruct MSParam;
    IritUserMicroRegularParamStruct *MSRegularParam;
    Cross4LocalDataStruct LclData;

    if (!read_fixed_dims(&LclData.NX, &LclData.NY, &LclData.NZ)) {
        fprintf(stderr, "Failed to read fixed dims.\n");
        return;
    }

    if (!read_shell_thicknesses(LclData.NX, LclData.NY, LclData.NZ,
                                &LclData.ShellThicknesses)) {
        fprintf(stderr, "Failed to read shell thicknesses.\n");
        return;
    }

    DefMapPObj = IritPrsrGetDataFiles(&InputDefMap, 1, FALSE, FALSE);
    if (DefMapPObj == NULL) {
        fprintf(stderr, "Failed to load the deformation function.\n");
        IritFree(LclData.ShellThicknesses);
        return;
    }
    assert(IRIT_PRSR_IS_TRIVAR_OBJ(DefMapPObj));
    TVMap = DefMapPObj->U.Trivars;
    DeformMV = IritMvarCnvrtTVToMV(TVMap);

    IRIT_ZAP_MEM(&MSParam, sizeof(IritUserMicroParamStruct));
    MSParam.TilingType = IRIT_USER_MICRO_TILE_REGULAR;
    MSParam.DeformMV = DeformMV;
    MSParam.ApproxLowOrder = 4;
    MSParam.ShellCapBits = 0;

    MSRegularParam = &MSParam.U.RegularParam;
    MSRegularParam->Tile = NULL;          /* Tile is synthesized on the fly. */
    MSRegularParam->TilingStepMode = TRUE;
    MSRegularParam->MaxPolyEdgeLen = 0.1;

    for (i = 0; i < 3; ++i) {
        MSRegularParam->TilingSteps[i].TilesPerIntervals =
            (CagdRType *)IritMalloc(sizeof(CagdRType) * 2);
        MSRegularParam->TilingSteps[i].Len = 1;
    }

    MSRegularParam->TilingSteps[0].TilesPerIntervals[0] = LclData.NX;
    MSRegularParam->TilingSteps[1].TilesPerIntervals[0] = LclData.NY;
    MSRegularParam->TilingSteps[2].TilesPerIntervals[0] = LclData.NZ;

    MSRegularParam->PreProcessCBFunc = PreProcessTile;
    MSRegularParam->CBFuncData = &LclData;

    MS = IritUserMicroStructComposition(&MSParam);
    if (MS == NULL) {
        fprintf(stderr, "IritUserMicroStructComposition returned NULL.\n");
        IritFree(LclData.ShellThicknesses);
        return;
    }

    /* Compute volume by summing all trivariates, regardless of whether the   */
    /* composition returned a list of tiles or a single bare trivariate.      */
    {
        CagdRType volume = SumTrivarVolumes(MS);

        FILE *fp = fopen("props.txt", "w");
        if (fp == NULL) {
            perror("Failed to open props.txt");
            IritFree(LclData.ShellThicknesses);
            return;
        }
        fprintf(fp, "volume %f\n", volume);
        fclose(fp);
    }

    Handler = IritPrsrOpenDataFile("model.itd", FALSE, 1);
    if (MS != NULL) {
        IritPrsrPutObjectToHandler(Handler, MS);
        IritPrsrFreeObject(MS);
    }
    IritPrsrCloseStream(Handler, TRUE);

    IritMvarMVFree(DeformMV);
    IritUserMicroTileFree(MSRegularParam->Tile);

    for (i = 0; i < 3; ++i)
        IritFree(MSRegularParam->TilingSteps[i].TilesPerIntervals);

    IritTrivTVFree(TVMap);
    IritFree(LclData.ShellThicknesses);
}

int main(int argc, char **argv)
{
    GenerateMicroStructures();
    return 0;
}
