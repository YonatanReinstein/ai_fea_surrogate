/*****************************************************************************
*   Constructs locally varying trivariate tiles in microstructure            *
* constructions using a BiStableCollapse2DXY2 tile.                          *
******************************************************************************
* (C) Gershon Elber, Technion, Israel Institute of Technology                *
******************************************************************************
* Written by:  Gershon Elber                            Ver 1.0, Dec 2017    *
* Adapted for BiStableCollapse2DXY2 tile                                     *
*****************************************************************************/

#include "inc_irit/irit_sm.h"
#include "inc_irit/iritprsr.h"
#include "inc_irit/allocate.h"
#include "inc_irit/attribut.h"
#include "inc_irit/geom_lib.h"
#include "inc_irit/cagd_lib.h"
#include "inc_irit/user_lib.h"
#include "inc_irit/grap_lib.h"

#define BISTABLE_JOINTS_SIZE     0.7
#define BISTABLE_FLEX_CLIP_RATIO 0.3

typedef struct BistableLocalDataStruct {
    int NX, NY, NZ;
    CagdRType *FrameThicknesses;  /* NX*NY*NZ values, X-outermost order */
} BistableLocalDataStruct;

static int read_bistable_params(int *NX, int *NY, int *NZ,
                                CagdRType **FrameThicknesses);
static IritPrsrObjectStruct *PreProcessTile(IritPrsrObjectStruct *Tile,
    IritUserMicroPreProcessTileCBStruct *CBData);
static void GenerateMicroStructures(void);

/*****************************************************************************
* DESCRIPTION:
*   Read tiling grid dimensions and per-tile FrameThickness values from
*   dims.itd.
*   List format: [NX, NY, NZ, ft_0, ft_1, ..., ft_{NX*NY*NZ-1}]
*   Values are stored in X-outermost (U-outermost) order.
*
* RETURN VALUE:
*   int: TRUE on success, FALSE on failure.
*****************************************************************************/
static int read_bistable_params(int *NX, int *NY, int *NZ,
                                CagdRType **FrameThicknesses)
{
    IritPrsrObjectStruct *PObj;
    const char *DimsFileName = "dims.itd";
    int i, n;

    PObj = IritPrsrGetDataFiles(&DimsFileName, 1, FALSE, FALSE);
    if (PObj == NULL) {
        fprintf(stderr, "Failed to load dims.itd\n");
        return FALSE;
    }

    if (!IRIT_PRSR_IS_NUM_OBJ(PObj)) { fprintf(stderr, "NX missing\n"); return FALSE; }
    *NX = (int)PObj->U.R;
    PObj = PObj->Pnext;

    if (PObj == NULL || !IRIT_PRSR_IS_NUM_OBJ(PObj)) { fprintf(stderr, "NY missing\n"); return FALSE; }
    *NY = (int)PObj->U.R;
    PObj = PObj->Pnext;

    if (PObj == NULL || !IRIT_PRSR_IS_NUM_OBJ(PObj)) { fprintf(stderr, "NZ missing\n"); return FALSE; }
    *NZ = (int)PObj->U.R;
    PObj = PObj->Pnext;

    n = (*NX) * (*NY) * (*NZ);
    *FrameThicknesses = (CagdRType *)IritMalloc(sizeof(CagdRType) * n);

    for (i = 0; i < n; i++) {
        if (PObj == NULL || !IRIT_PRSR_IS_NUM_OBJ(PObj)) {
            fprintf(stderr, "FrameThickness[%d] missing\n", i);
            IritFree(*FrameThicknesses);
            return FALSE;
        }
        (*FrameThicknesses)[i] = PObj->U.R;
        PObj = PObj->Pnext;
    }

    return TRUE;
}

/*****************************************************************************
* DESCRIPTION:
*   Callback invoked for each tile in the 3D grid.  Looks up the
*   FrameThickness for this tile by its (ix, iy, iz) index, then constructs
*   a BiStableCollapse2DXY2 tile and maps it through the deformation matrix.
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
    BistableLocalDataStruct
        *LclData = (BistableLocalDataStruct *)CBData->CBFuncData;
    int ix = CBData->TileIdxs[0];
    int iy = CBData->TileIdxs[1];
    int iz = CBData->TileIdxs[2];
    int idx = ix * LclData->NY * LclData->NZ + iy * LclData->NZ + iz;
    CagdRType FrameThickness = LclData->FrameThicknesses[idx];
    CagdBType FlexArms[4] = {TRUE, TRUE, TRUE, TRUE};
    char *Error = NULL;

    assert(Tile == NULL);

    Tile = IritUserMicroBiStableCollapse2DXY2(
        FrameThickness,
        BISTABLE_JOINTS_SIZE,
        BISTABLE_FLEX_CLIP_RATIO,
        FlexArms,
        0,
        &Error);

    if (Tile == NULL) {
        fprintf(stderr, "IritUserMicroBiStableCollapse2DXY2 failed at [%d,%d,%d]: %s\n",
                ix, iy, iz, Error ? Error : "unknown error");
        return NULL;
    }

    Tile = IritGeomTransformObjectInPlace(Tile, CBData->Mat);

    return Tile;
}

/*****************************************************************************
* DESCRIPTION:
*   Create a micro structure tiled with BiStableCollapse2DXY2 tiles.
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
    BistableLocalDataStruct LclData;

    if (!read_bistable_params(&LclData.NX, &LclData.NY, &LclData.NZ,
                              &LclData.FrameThicknesses)) {
        fprintf(stderr, "Failed to read bistable parameters.\n");
        return;
    }

    DefMapPObj = IritPrsrGetDataFiles(&InputDefMap, 1, FALSE, FALSE);
    if (DefMapPObj == NULL) {
        fprintf(stderr, "Failed to load the deformation function.\n");
        IritFree(LclData.FrameThicknesses);
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

    /* Compute volume by summing all trivariates across all tiles. */
    {
        int tiles_num = LclData.NX * LclData.NY * LclData.NZ;
        CagdRType volume = 0.0;
        int index;

        for (index = 0; index < tiles_num; index++) {
            IritPrsrObjectStruct *MV = MS->U.Lst.PObjList[index];
            int j;
            for (j = 0; MV->U.Lst.PObjList[j] != NULL; j++) {
                IritPrsrObjectStruct *MQ = MV->U.Lst.PObjList[j];
                if (IRIT_PRSR_IS_TRIVAR_OBJ(MQ)) {
                    IritTrivTVStruct *TV;
                    for (TV = MQ->U.Trivars; TV != NULL; TV = TV->Pnext)
                        volume += fabs(IritTrivTVVolume(TV, TRUE));
                }
            }
        }

        {
            FILE *fp = fopen("props.txt", "w");
            if (fp == NULL) {
                perror("Failed to open props.txt");
                IritFree(LclData.FrameThicknesses);
                return;
            }
            fprintf(fp, "volume %f\n", volume);
            fclose(fp);
        }
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
    IritFree(LclData.FrameThicknesses);
}

int main(int argc, char **argv)
{
    GenerateMicroStructures();
    return 0;
}
