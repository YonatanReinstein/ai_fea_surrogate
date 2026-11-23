/*****************************************************************************
*   Constructs locally varying trivariate tiles in microstructure            *
* constructions using a hollow-cube tile open along Z.                       *
*                                                                            *
*   Each tile is a square tube: four box-trivariate walls (the X- and        *
* Y-perpendicular cube faces) of a per-tile ShellThickness.  The two faces   *
* perpendicular to Z are left open, so the tile is hollow and open along Z.  *
******************************************************************************
* (C) Gershon Elber, Technion, Israel Institute of Technology                *
******************************************************************************
* Written by:  Gershon Elber                            Ver 1.0, Dec 2017    *
* Adapted for a hollow-cube tile (open along Z)                              *
*****************************************************************************/

#include <stdlib.h>
#include "inc_irit/irit_sm.h"
#include "inc_irit/iritprsr.h"
#include "inc_irit/allocate.h"
#include "inc_irit/attribut.h"
#include "inc_irit/geom_lib.h"
#include "inc_irit/cagd_lib.h"
#include "inc_irit/triv_lib.h"
#include "inc_irit/user_lib.h"
#include "inc_irit/grap_lib.h"

/* Cap the effective wall thickness just below 0.5.  At exactly 0.5 the inner */
/* boundary collapses to the tile center, turning the trapezoidal walls into  */
/* triangular prisms whose meshed hex elements are degenerate (zero volume,   */
/* infinite aspect ratio) and are rejected by the FE solver.  Keeping a small */
/* margin leaves a valid near-solid cell instead.                             */
#define HOLLOW_CUBE_MAX_THICKNESS 0.48

typedef struct HollowCubeLocalDataStruct {
    int NX, NY, NZ;
    CagdRType *ShellThicknesses;  /* NX*NY*NZ values, X-outermost order */
} HollowCubeLocalDataStruct;

static int read_fixed_dims(int *NX, int *NY, int *NZ);
static int read_shell_thicknesses(int NX, int NY, int NZ,
                                  CagdRType **ShellThicknesses);
static IritPrsrObjectStruct *BuildHollowCubeTile(CagdRType ShellThickness);
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
*   Rotate the planar point (x, y) by k * 90 degrees counter-clockwise about
*   the tile center (0.5, 0.5), keeping z, and store it in P.
*****************************************************************************/
static void RotateAboutCenter(CagdRType x, CagdRType y, CagdRType z,
                              int k, CagdPType P)
{
    int i;

    for (i = 0; i < k; i++) {           /* (x, y) -> (1 - y, x) per 90 deg. */
        CagdRType nx = 1.0 - y,
                  ny = x;

        x = nx;
        y = ny;
    }

    P[0] = x;
    P[1] = y;
    P[2] = z;
}

/*****************************************************************************
* DESCRIPTION:
*   Build a hollow-cube tile in the unit cube [0, 1]^3 from four identical
*   trapezoidal-prism walls.  The square cross-section frame (outer boundary
*   [0, 1]^2, inner boundary [t, 1 - t]^2) is split along its diagonals into
*   four congruent trapezoids; each is extruded along the full Z extent
*   [0, 1], leaving the two Z-perpendicular faces open.
*
*   In every wall the U = 0 face sits on the outer tile boundary and the
*   U = 1 face on the inner boundary, so each trivariate runs from the outer
*   corners to the inner corners.  The four walls are the same trapezoid
*   rotated by 0, 90, 180 and 270 degrees about the tile center.
*
* PARAMETERS:
*   t:  Shell (wall) thickness, in (0.0, 0.5].
*
* RETURN VALUE:
*   IritPrsrObjectStruct *: A LIST object holding the four wall trivariates,
*                           or NULL on bad input.
*****************************************************************************/
/*****************************************************************************
* DESCRIPTION:
*   Rotate every control point of TV by k * 90 degrees about the tile centre *
*   (x, y) -> (1 - y, x), as RotateAboutCenter does for points.  Works on    *
*   rational control points too (homogeneous: X' = w - Y, Y' = X).           *
*****************************************************************************/
static void RotateTVAboutCenter(IritTrivTVStruct *TV, int k)
{
    int i, j,
        n = TV -> ULength * TV -> VLength * TV -> WLength;
    CagdRType *W = TV -> Points[0],
        *X = TV -> Points[1],
        *Y = TV -> Points[2];

    for (j = 0; j < k; j++)
        for (i = 0; i < n; i++) {
            CagdRType w = W != NULL ? W[i] : 1.0,
                nx = w - Y[i],
                ny = X[i];

            X[i] = nx;
            Y[i] = ny;
        }
}

/*****************************************************************************
* DESCRIPTION:
*   Inner edge of the Y-min wall of a tile whose void has corners rounded
*   with radius r, as the graph y = f(x), x in [a, 1 - a]:
*     arc around (t + r, t + r) for x < t + r, the straight side y = t, and
*     the mirrored arc for x > 1 - t - r;  a = t + r (1 - 1 / sqrt 2) is
*     where the arc meets the tile diagonal.  r = 0 gives the sharp edge.
*****************************************************************************/
static CagdRType InnerEdgeY(CagdRType x, CagdRType t, CagdRType r)
{
    CagdRType c, d;

    if (x > 0.5)
        x = 1.0 - x;                                     /* Symmetric. */
    c = t + r;
    if (r <= 0.0 || x >= c)
        return t;
    d = c - x;
    return c - sqrt(IRIT_MAX(r * r - d * d, 0.0));
}

/*****************************************************************************
* DESCRIPTION:
*   The Y-min wall (before rotation by k * 90 degrees) of a tile whose void
*   has rounded corners of radius r, as THREE trivariates side by side.
*   Each is ruled between a third of the straight outer edge, x in
*   [i/3, (i+1)/3], and the matching third of the inner edge's own extent,
*   x in [a + i (1 - 2a)/3, a + (i+1)(1 - 2a)/3], then extruded along Z.
*   Both cuts are at fixed fractions, so
*     - every tile's outer faces carry the same node layout whatever its t
*       and r (meshes conform across tiles), and
*     - each outer third faces the inner third right above it, so the
*       rulings stay near-perpendicular to the wall for ANY radius, r = 0
*       included (cutting the inner edge at the arc ends instead folded the
*       elements of thin walls with small radii).
*   Each inner third is one cubic Bezier: end points on the exact edge
*   (InnerEdgeY), the two inner control points a least-squares fit to 41
*   samples of it, so neighbouring thirds meet exactly and the diagonal
*   ends sit at (a, a) / (1 - a, a) where the rotated walls expect them.
*   U runs outer -> inner, V along the wall, W along Z, as in the box walls.
*****************************************************************************/
static IritTrivTVStruct *RoundedWall(CagdRType t, CagdRType r, int k)
{
    int i, j, m;
    CagdRType a = t + r * (1.0 - 1.0 / sqrt(2.0)),
        Span = (1.0 - 2.0 * a) / 3.0;
    IritTrivTVStruct *First = NULL, *Last = NULL;

    for (i = 0; i < 3; i++) {
        CagdRType xi0 = a + i * Span, xi1 = a + (i + 1) * Span,
            xo0 = i / 3.0, xo1 = (i + 1) / 3.0,
            P0[2], P3[2], A11 = 0.0, A12 = 0.0, A22 = 0.0, R1[2] = { 0, 0 },
            R2[2] = { 0, 0 }, Det, Q1[2], Q2[2];
        IritCagdCrvStruct *Inner = IritCagdBzrCrvNew(4, IRIT_CAGD_PT_E3_TYPE),
            *Outer = IritCagdBzrCrvNew(4, IRIT_CAGD_PT_E3_TYPE);
        IritCagdSrfStruct *Srf, *Srf2;
        IritCagdVecStruct Dir;
        IritTrivTVStruct *TV;

        P0[0] = xi0; P0[1] = InnerEdgeY(xi0, t, r);
        P3[0] = xi1; P3[1] = InnerEdgeY(xi1, t, r);
        /* Least squares for the two free control points, samples at     */
        /* uniform u <-> uniform x (the edge is a graph over x).           */
        for (m = 0; m <= 40; m++) {
            CagdRType u = m / 40.0,
                b0 = (1 - u) * (1 - u) * (1 - u), b1 = 3 * u * (1 - u) * (1 - u),
                b2 = 3 * u * u * (1 - u), b3 = u * u * u,
                X = xi0 + u * (xi1 - xi0), S[2];

            S[0] = X - b0 * P0[0] - b3 * P3[0];
            S[1] = InnerEdgeY(X, t, r) - b0 * P0[1] - b3 * P3[1];
            A11 += b1 * b1; A12 += b1 * b2; A22 += b2 * b2;
            for (j = 0; j < 2; j++) {
                R1[j] += b1 * S[j];
                R2[j] += b2 * S[j];
            }
        }
        Det = A11 * A22 - A12 * A12;
        for (j = 0; j < 2; j++) {
            Q1[j] = (A22 * R1[j] - A12 * R2[j]) / Det;
            Q2[j] = (A11 * R2[j] - A12 * R1[j]) / Det;
        }
        Inner -> Points[1][0] = P0[0]; Inner -> Points[2][0] = P0[1];
        Inner -> Points[1][1] = Q1[0]; Inner -> Points[2][1] = Q1[1];
        Inner -> Points[1][2] = Q2[0]; Inner -> Points[2][2] = Q2[1];
        Inner -> Points[1][3] = P3[0]; Inner -> Points[2][3] = P3[1];
        for (m = 0; m < 4; m++) {
            Inner -> Points[3][m] = 0.0;
            Outer -> Points[1][m] = xo0 + (xo1 - xo0) * m / 3.0;
            Outer -> Points[2][m] = 0.0;
            Outer -> Points[3][m] = 0.0;
        }

        Srf = IritCagdRuledSrf(Outer, Inner, 2, 2);   /* u along, v outer -> inner */
        Srf2 = IritCagdSrfReverse2(Srf);               /* u outer -> inner, v along */
        Dir.Vec[0] = 0.0; Dir.Vec[1] = 0.0; Dir.Vec[2] = 1.0;
        TV = IritTrivExtrudeTV(Srf2, &Dir);
        IritCagdSrfFree(Srf);
        IritCagdSrfFree(Srf2);
        IritCagdCrvFree(Inner);
        IritCagdCrvFree(Outer);
        RotateTVAboutCenter(TV, k);
        TV -> Pnext = NULL;
        if (First == NULL)
            First = TV;
        else
            Last -> Pnext = TV;
        Last = TV;
    }
    return First;
}

static IritPrsrObjectStruct *BuildHollowCubeTile(CagdRType t)
{
    int k;
    IritTrivTVStruct *Walls[4];
    IritPrsrObjectStruct *RetObj, *PObj;
    CagdPType P000, P001, P010, P011, P100, P101, P110, P111;
    /* HOLLOW_CUBE_CORNER_FRAC: round the void's corners with radius        */
    /*   r = frac * (0.5 - t) * (1 - exp(-(t - tmin) / w)),                  */
    /* a fraction of the void's half-width (frac 1 = a circular void),      */
    /* faded smoothly to ~0 at the floor tmin so a floor tile is a bare     */
    /* thin tube (no corner material the optimizer cannot remove).          */
    /* w = HOLLOW_CUBE_CORNER_RAMP (default 0.05), tmin =                   */
    /* HOLLOW_CUBE_CORNER_TMIN (default 0.01, dims.json's floor).  Every  */
    /* tile meshes in the same layout (3 pieces per wall, RoundedWall)      */
    /* whatever its t and r, r = 0 included.  Unset or 0: the original     */
    /* sharp-cornered box walls.                                            */
    const char *FracStr = getenv("HOLLOW_CUBE_CORNER_FRAC"),
        *RampStr = getenv("HOLLOW_CUBE_CORNER_RAMP"),
        *TMinStr = getenv("HOLLOW_CUBE_CORNER_TMIN");
    CagdRType Frac = FracStr != NULL && FracStr[0] != 0 ? atof(FracStr) : 0.0,
        Ramp = RampStr != NULL && RampStr[0] != 0 ? atof(RampStr) : 0.05,
        TMin = TMinStr != NULL && TMinStr[0] != 0 ? atof(TMinStr) : 0.01;

    if (t <= 0.0 || t > 0.5)
        return NULL;
    if (t > HOLLOW_CUBE_MAX_THICKNESS)   /* Avoid the degenerate t = 0.5 cell. */
        t = HOLLOW_CUBE_MAX_THICKNESS;

    if (Frac > 0.0) {
        CagdRType Fade = Ramp > 0.0 ? 1.0 - exp(-IRIT_MAX(t - TMin, 0.0) / Ramp) : 1.0,
            r = IRIT_MIN(Frac, 1.0) * (0.5 - t) * Fade;
        IritTrivTVStruct *All = NULL, *Tail = NULL, *W;

        for (k = 0; k < 4; k++) {               /* 3 trivariates per wall */
            W = RoundedWall(t, r, k);
            if (All == NULL)
                All = W;
            else
                Tail -> Pnext = W;
            for (Tail = W; Tail -> Pnext != NULL; Tail = Tail -> Pnext);
        }
        RetObj = IritPrsrGenLISTObject(PObj = IritPrsrGenTRIVARObject(All));
        IritMiscAttrIDSetObjectRGBColor(PObj, 255, 0, 0);
        return RetObj;
    }

    /* Base trapezoid (the Y-min wall), then its 90-degree rotations.  The    */
    /* U = 0 (P0xx) edge spans the outer boundary corners (0,0)-(1,0); the    */
    /* U = 1 (P1xx) edge spans the inner boundary corners (t,t)-(1-t,t).      */
    for (k = 0; k < 4; k++) {
        RotateAboutCenter(0.0,     0.0, 0.0, k, P000); /* outer corner A, z=0 */
        RotateAboutCenter(0.0,     0.0, 1.0, k, P001); /* outer corner A, z=1 */
        RotateAboutCenter(1.0,     0.0, 0.0, k, P010); /* outer corner B, z=0 */
        RotateAboutCenter(1.0,     0.0, 1.0, k, P011); /* outer corner B, z=1 */
        RotateAboutCenter(t,       t,   0.0, k, P100); /* inner corner A, z=0 */
        RotateAboutCenter(t,       t,   1.0, k, P101); /* inner corner A, z=1 */
        RotateAboutCenter(1.0 - t, t,   0.0, k, P110); /* inner corner B, z=0 */
        RotateAboutCenter(1.0 - t, t,   1.0, k, P111); /* inner corner B, z=1 */

        Walls[k] = IritTrivNSPrimGenBox(P000, P001, P010, P011,
                                        P100, P101, P110, P111);
    }

    for (k = 0; k < 3; k++)
        Walls[k] -> Pnext = Walls[k + 1];
    Walls[3] -> Pnext = NULL;

    RetObj = IritPrsrGenLISTObject(PObj = IritPrsrGenTRIVARObject(Walls[0]));
    IritMiscAttrIDSetObjectRGBColor(PObj, 255, 0, 0);

    return RetObj;
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
    HollowCubeLocalDataStruct
        *LclData = (HollowCubeLocalDataStruct *)CBData->CBFuncData;
    int ix = CBData->TileIdxs[0];
    int iy = CBData->TileIdxs[1];
    int iz = CBData->TileIdxs[2];
    int idx = ix * LclData->NY * LclData->NZ + iy * LclData->NZ + iz;
    CagdRType ShellThickness = LclData->ShellThicknesses[idx];

    assert(Tile == NULL);

    Tile = BuildHollowCubeTile(ShellThickness);

    if (Tile == NULL) {
        fprintf(stderr, "BuildHollowCubeTile failed at [%d,%d,%d] (t=%g)\n",
                ix, iy, iz, ShellThickness);
        return NULL;
    }

    Tile = IritGeomTransformObjectInPlace(Tile, CBData->Mat);

    return Tile;
}

/*****************************************************************************
* DESCRIPTION:
*   Create a micro structure tiled with hollow-cube tiles.
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
    HollowCubeLocalDataStruct LclData;

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
