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
static IritPrsrObjectStruct *BuildHollowCubeTile(CagdRType t)
{
    int k;
    IritTrivTVStruct *Walls[4];
    IritPrsrObjectStruct *RetObj, *PObj;
    CagdPType P000, P001, P010, P011, P100, P101, P110, P111;

    if (t <= 0.0 || t > 0.5)
        return NULL;
    if (t > HOLLOW_CUBE_MAX_THICKNESS)   /* Avoid the degenerate t = 0.5 cell. */
        t = HOLLOW_CUBE_MAX_THICKNESS;

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
