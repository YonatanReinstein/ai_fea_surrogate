/*****************************************************************************
*   Constructs locally varying trivariate tiles	in microstructure	     *
* constructions	using a	call back function.  The wing DefMap is	one example. *
******************************************************************************
* (C) Gershon Elber, Technion, Israel Institute	of Technology		     *
******************************************************************************
* Written by:  Gershon Elber				Ver 1.0, Dec 2017    *
*****************************************************************************/

#include "face_centers.h"

typedef CagdRType(*LclThicknessFuncCBType)(CagdRType u,
    CagdRType v,
    CagdRType w);

typedef struct UserMicroLocalDataStruct { /* User specific data in CB funcs. */
    IritTrivTVStruct* DefMap;
    CagdRType BndryThickness;
    LclThicknessFuncCBType ThicknessFuncCB;
} UserMicroLocalDataStruct;

int g_ni, g_nj, g_nk;
CagdRType *g_data;


static void read_dims(void);

static int PreProcessTile1FaceParam(
    const IritUserMicroPreProcessTileCBStruct* CBData,
    IritTrivTVBndryType Bndry,
    CagdRType BndryThickness,
    LclThicknessFuncCBType ThicknessFuncCB,
    IritUserMicroTileBndryPrmStruct* BPrm);
static IritPrsrObjectStruct* PreProcessTile(IritPrsrObjectStruct* Tile,
    IritUserMicroPreProcessTileCBStruct* CBData);
static void GenerateMicroStructures(void);


/*****************************************************************************
* DESCRIPTION:
*   Prepare one	face parameters	for the	3D grid	tile synthesized on the	fly.
*
* PARAMETERS:
*   LclMinDmn, LclMaxDmn:     UVW domain of this tile, in the parent
*		deformation function.
*   Bndry:	THe boundary (out of UMin/Max, VMin/Max, WMin/Max).
*   BndryThickness:  To	set the	thickness of the synthesized boundary,
*		or 0.0 to disable.
*   ThicknessFuncCB: Call back function	to prescribe the thickness desired
*		based upon the UVW location.
*   BPrm:	Parameters of this face	to update.
*
* RETURN VALUE:
*   int:
*****************************************************************************/
static int PreProcessTile1FaceParam(
    const IritUserMicroPreProcessTileCBStruct* CBData,
    IritTrivTVBndryType Bndry,
    CagdRType BndryThickness,
    LclThicknessFuncCBType ThicknessFuncCB,
    IritUserMicroTileBndryPrmStruct* BPrm)
{
    int i;
    CagdRType u, v, w;
    const CagdRType
        * LclMinDmn = CBData->TileLclDmnMin,
        * LclMaxDmn = CBData->TileLclDmnMax,
        * DefMapDmnMin = CBData->DefMapDmnMin,
        * DefMapDmnMax = CBData->DefMapDmnMax;

    IRIT_ZAP_MEM(BPrm, sizeof(IritUserMicroTileBndryPrmStruct));

    switch (Bndry) {
    case IRIT_TRIV_U_MIN_BNDRY:
        u = LclMinDmn[0];
        v = (LclMinDmn[1] + LclMaxDmn[1]) * 0.5;
        w = (LclMinDmn[2] + LclMaxDmn[2]) * 0.5;
        break;
    case IRIT_TRIV_U_MAX_BNDRY:
        u = LclMaxDmn[0];
        v = (LclMinDmn[1] + LclMaxDmn[1]) * 0.5;
        w = (LclMinDmn[2] + LclMaxDmn[2]) * 0.5;
        break;
    case IRIT_TRIV_V_MIN_BNDRY:
        u = (LclMinDmn[0] + LclMaxDmn[0]) * 0.5;
        v = LclMinDmn[1];
        w = (LclMinDmn[2] + LclMaxDmn[2]) * 0.5;
        break;
    case IRIT_TRIV_V_MAX_BNDRY:
        u = (LclMinDmn[0] + LclMaxDmn[0]) * 0.5;
        v = LclMaxDmn[1];
        w = (LclMinDmn[2] + LclMaxDmn[2]) * 0.5;
        break;
    case IRIT_TRIV_W_MIN_BNDRY:
        u = (LclMinDmn[0] + LclMaxDmn[0]) * 0.5;
        v = (LclMinDmn[1] + LclMaxDmn[1]) * 0.5;
        w = LclMinDmn[2];
        break;
    case IRIT_TRIV_W_MAX_BNDRY:
        u = (LclMinDmn[0] + LclMaxDmn[0]) * 0.5;
        v = (LclMinDmn[1] + LclMaxDmn[1]) * 0.5;
        w = LclMaxDmn[2];
        break;
    default:
        break;
    }

    u = DefMapDmnMin[0] + u * (DefMapDmnMax[0] - DefMapDmnMin[0]);
    v = DefMapDmnMin[1] + v * (DefMapDmnMax[1] - DefMapDmnMin[1]);
    w = DefMapDmnMin[2] + w * (DefMapDmnMax[2] - DefMapDmnMin[2]);

    BPrm->BndryShape = 0.25;
    BPrm->OuterRadius = ThicknessFuncCB(u, v, w);

    switch (Bndry) {
    case IRIT_TRIV_U_MIN_BNDRY:
    case IRIT_TRIV_U_MAX_BNDRY:
        BPrm->Bndry[0] = ThicknessFuncCB(u, LclMinDmn[1], LclMinDmn[2]);
        BPrm->Bndry[1] = ThicknessFuncCB(u, LclMaxDmn[1], LclMinDmn[2]);
        BPrm->Bndry[2] = ThicknessFuncCB(u, LclMinDmn[1], LclMaxDmn[2]);
        BPrm->Bndry[3] = ThicknessFuncCB(u, LclMaxDmn[1], LclMaxDmn[2]);
        break;
    case IRIT_TRIV_V_MIN_BNDRY:
    case IRIT_TRIV_V_MAX_BNDRY:
        BPrm->Bndry[0] = ThicknessFuncCB(LclMinDmn[0], v, LclMinDmn[2]);
        BPrm->Bndry[1] = ThicknessFuncCB(LclMaxDmn[0], v, LclMinDmn[2]);
        BPrm->Bndry[2] = ThicknessFuncCB(LclMinDmn[0], v, LclMaxDmn[2]);
        BPrm->Bndry[3] = ThicknessFuncCB(LclMaxDmn[0], v, LclMaxDmn[2]);
        break;
    case IRIT_TRIV_W_MIN_BNDRY:
    case IRIT_TRIV_W_MAX_BNDRY:
        BPrm->Bndry[0] = ThicknessFuncCB(LclMinDmn[0], LclMinDmn[1], w);
        BPrm->Bndry[1] = ThicknessFuncCB(LclMaxDmn[0], LclMinDmn[1], w);
        BPrm->Bndry[2] = ThicknessFuncCB(LclMinDmn[0], LclMaxDmn[1], w);
        BPrm->Bndry[3] = ThicknessFuncCB(LclMaxDmn[0], LclMaxDmn[1], w);
        break;
    default:
        break;
    }

    for (i = 0; i < 4; i++)
        BPrm->Bndry[i] *= BndryThickness;

    return TRUE;
}

/*****************************************************************************
* DESCRIPTION:
*
*
*
* PARAMETERS:
*   Tile:   Tile to preprocess.	 Here we expect	a NULL as we build tile	     *
*	    from scratch.
*   CBData: The	call back data.
*
* RETURN VALUE:
*   IPObjectStruct *:
*****************************************************************************/
static IritPrsrObjectStruct* PreProcessTile(IritPrsrObjectStruct* Tile,
    IritUserMicroPreProcessTileCBStruct* CBData)
{
    UserMicroLocalDataStruct
        * LclData = (UserMicroLocalDataStruct*)CBData->CBFuncData;
    CagdRType UMin, UMax, VMin, VMax, WMin, WMax,
        * LclMinDmn = CBData->TileLclDmnMin,
        * LclMaxDmn = CBData->TileLclDmnMax,
        BndryThickness = LclData->BndryThickness;
    IritGeomBBBboxStruct BBox;
    LclThicknessFuncCBType
        ThicknessFuncCB = LclData->ThicknessFuncCB;
    IritUserMicroTileBndryPrmStruct UMinPrms, UMaxPrms,
        VMinPrms, VMaxPrms,
        WMinPrms, WMaxPrms;

    assert(Tile == NULL);                          /* We build tiles here... */

    IritTrivTVDomain(LclData->DefMap, &UMin, &UMax, &VMin, &VMax, &WMin, &WMax);

    //fprintf(stderr, "Tile[%d,%d,%d] from (%.3f, %.3f %.3f) to (%.3f, %.3f, %.3f)\r\n",
    //    CBData->TileIdxs[0],
    //    CBData->TileIdxs[1],
    //    CBData->TileIdxs[2],
    //    LclMinDmn[0],
    //    LclMinDmn[1],
    //    LclMinDmn[2],
    //    LclMaxDmn[0],
    //    LclMaxDmn[1],
    //    LclMaxDmn[2]);

    if (!PreProcessTile1FaceParam(CBData, IRIT_TRIV_U_MIN_BNDRY,
        0.0, ThicknessFuncCB, &UMinPrms) ||
        !PreProcessTile1FaceParam(CBData, IRIT_TRIV_U_MAX_BNDRY,
            0.0, ThicknessFuncCB, &UMaxPrms) ||
        !PreProcessTile1FaceParam(CBData, IRIT_TRIV_V_MIN_BNDRY,
            0.0, ThicknessFuncCB, &VMinPrms) ||
        !PreProcessTile1FaceParam(CBData, IRIT_TRIV_V_MAX_BNDRY,
            0.0, ThicknessFuncCB, &VMaxPrms) ||
        !PreProcessTile1FaceParam(CBData, IRIT_TRIV_W_MIN_BNDRY,
            0.0, ThicknessFuncCB, &WMinPrms) ||
        !PreProcessTile1FaceParam(CBData, IRIT_TRIV_W_MAX_BNDRY,
            0.0, ThicknessFuncCB, &WMaxPrms)) {
        return NULL;
    }

    Tile = IritUserMicro3DCrossTile(&UMinPrms, &UMaxPrms, &VMinPrms, &VMaxPrms,
        &WMinPrms, &WMaxPrms, FALSE, NULL);
    IritGeomBBComputeBboxObject(Tile, &BBox, FALSE);
    if (BBox.Min[0] < 0.0 || BBox.Max[0] > 1.0 ||
        BBox.Min[1] < 0.0 || BBox.Max[1] > 1.0 ||
        BBox.Min[2] < 0.0 || BBox.Max[2] > 1.0) {
        fprintf(stderr, "Warning: Tile spans beyond the unit box.\n");
    }

#define DEBUG_VERIFY_JACOBIAN
#ifdef DEBUG_VERIFY_JACOBIAN
    {
        IritTrivTVStruct* TV;

        for (TV = Tile->U.Trivars; TV != NULL; TV = TV->Pnext) {
            IritMvarMVStruct
                * J = IritMvarCalculateTVJacobian(TV);
            CagdBBoxStruct BBox;

            IritMvarMVBBox(J, &BBox);
            IritMvarMVFree(J);
            if (BBox.Min[0] * BBox.Max[0] < 0.0) {
                fprintf(stderr, "Warning: Negative Jacobian tile found\n");
                //IritTrivDbg(TV);
            }
        }
    }
#endif /* DEBUG_VERIFY_JACOBIAN */

    Tile = IritGeomTransformObjectInPlace(Tile, CBData->Mat);

#define DEBUG_USER_MS_MAKE_TV_OBJS
#ifdef DEBUG_USER_MS_MAKE_TV_OBJS
    {
        IritPrsrObjectStruct* PTmp;
        IritTrivTVStruct* TV, * BTV,
            * BzrTVs = NULL;

        for (TV = Tile->U.Trivars; TV != NULL; TV = TV->Pnext) {
            if (IRIT_TRIV_IS_BEZIER_TV(TV)) {
                BTV = IritTrivTVCopy(TV);
                IRIT_LIST_PUSH(BTV, BzrTVs);
            }
            else
                BzrTVs = IritCagdListAppend(IritTrivCnvrtBsp2BzrTV(TV), BzrTVs);
        }
        PTmp = IritPrsrLnkListToListObject(BzrTVs, IRIT_PRSR_OBJ_TRIVAR);
        IritPrsrFreeObject(Tile);
        Tile = PTmp;
    }
#endif /* DEBUG_USER_MS_MAKE_TV_OBJS */

    return Tile;
}


static void read_dims(void)
{
    IritPrsrObjectStruct* PObj;
    const char* DimsFileName = "dims.itd";
    int i, n;

    PObj = IritPrsrGetDataFiles(&DimsFileName, 1, FALSE, FALSE);
    if (PObj == NULL) {
        fprintf(stderr, "Failed to load dims.itd\n");
        exit(1);
    }

    if (!IRIT_PRSR_IS_NUM_OBJ(PObj)) {
        fprintf(stderr, "First object is missing or not numeric\n");
        exit(1);
    }
    g_ni = (int)PObj->U.R;
    PObj = PObj->Pnext;

    if (PObj == NULL || !IRIT_PRSR_IS_NUM_OBJ(PObj)) {
        fprintf(stderr, "Second object is missing or not numeric\n");
        exit(1);
    }
    g_nj = (int)PObj->U.R;
    PObj = PObj->Pnext;

    if (PObj == NULL || !IRIT_PRSR_IS_NUM_OBJ(PObj)) {
        fprintf(stderr, "Third object is missing or not numeric\n");
        exit(1);
    }
    g_nk = (int)PObj->U.R;
    PObj = PObj->Pnext;

    n = face_count(g_ni, g_nj, g_nk);
    g_data = (CagdRType*)IritMalloc(sizeof(CagdRType) * n);
    IRIT_ZAP_MEM(g_data, sizeof(CagdRType) * n);

    for (i = 0; i < n; i++) {
        if (PObj == NULL || !IRIT_PRSR_IS_NUM_OBJ(PObj)) {
            fprintf(stderr, "Invalid data at %d\n", i);
            exit(1);
        }
        g_data[i] = PObj->U.R;
        PObj = PObj->Pnext;
    }
}

/*****************************************************************************
* DESCRIPTION:
*   Create a micro structure with a varying-in-size tiling example.
*
* PARAMETERS:
*   None
*
* RETURN VALUE:
*   void
*****************************************************************************/
static void GenerateMicroStructures(void)
{
    const char* InputDefMap = "outline.itd";
    int i, Handler;
    IritPrsrObjectStruct* MS, * DefMapPObj;
    IritMvarMVStruct* DeformMV;
    IritTrivTVStruct* TVMap;
    IritUserMicroParamStruct MSParam;
    IritUserMicroRegularParamStruct* MSRegularParam;
    UserMicroLocalDataStruct LclData;
    //printf("Loading deformation function...\n");

    DefMapPObj = IritPrsrGetDataFiles(&InputDefMap, 1, FALSE, FALSE);
    if (DefMapPObj == NULL) {
        fprintf(stderr, "Failed to load the deformation function.\n");
        return;
    }
    assert(IRIT_PRSR_IS_TRIVAR_OBJ(DefMapPObj));
    TVMap = DefMapPObj->U.Trivars;
    DeformMV = IritMvarCnvrtTVToMV(TVMap);

    /* Create the structure to be passed to the callback function. */
    LclData.DefMap = TVMap;
    LclData.BndryThickness = 3;

    IRIT_ZAP_MEM(&MSParam, sizeof(IritUserMicroParamStruct));
    MSParam.TilingType = IRIT_USER_MICRO_TILE_REGULAR;
    MSParam.DeformMV = DeformMV;
    MSParam.ApproxLowOrder = 4;

    /* Sets boundary end conditions on the geometry - cap the tiles in all  */
    /* boundaries and as a side effect color trivar tiles on boundaries,    */
    /* so one can set boundary conditions (i.e. toward analysis).           */
    MSParam.ShellCapBits = 0;

    MSRegularParam = &MSParam.U.RegularParam;
    MSRegularParam->Tile = NULL;       /* Tile is synthesized on the fly. */
    MSRegularParam->TilingStepMode = TRUE;
    MSRegularParam->MaxPolyEdgeLen = 0.1;

    for (i = 0; i < 3; ++i) {
        MSRegularParam->TilingSteps[i].TilesPerIntervals = (CagdRType*)IritMalloc(sizeof(CagdRType) * 2);
        MSRegularParam->TilingSteps[i].Len = 1;
    }

    MSRegularParam->TilingSteps[0].TilesPerIntervals[0] = 3;
    MSRegularParam->TilingSteps[1].TilesPerIntervals[0] = 3;
    MSRegularParam->TilingSteps[2].TilesPerIntervals[0] = 3;

    /* Call back function - will be called for each tile in the grid just   */
    /* before it is mapped through the deformation function, with the tile  */
    /* (that can be modified) and call back data.                           */
    MSRegularParam->PreProcessCBFunc = PreProcessTile;
    MSRegularParam->CBFuncData = &LclData;         /* The call back data. */

    LclData.ThicknessFuncCB = UniformTilingCB;

    read_dims();
    MS = IritUserMicroStructComposition(&MSParam);
    int tiles_num = MSRegularParam->TilingSteps[0].TilesPerIntervals[0] * MSRegularParam->TilingSteps[1].TilesPerIntervals[0] * MSRegularParam->TilingSteps[2].TilesPerIntervals[0];
    int params_per_tile = 7;
    CagdRType volume = 0;
    int index = 0;
    while (index < tiles_num) {
        IritPrsrObjectStruct* MV = MS->U.Lst.PObjList[index];
        int j = 0;
        while (j < params_per_tile) {
            IritPrsrObjectStruct* MQ = MV->U.Lst.PObjList[j];
            volume += fabs(IritTrivTVVolume(MQ->U.Trivars, TRUE));
            j++;
        }
        index++;
    }

    FILE* fp = fopen("props.txt", "w");
    if (fp == NULL) {
        perror("Failed to open file");
        return;
    }
    fprintf(fp, "volume %f\n", volume);
    fclose(fp);

    Handler = IritPrsrOpenDataFile("model.itd", FALSE, 1);
    if (MS != NULL) {
        IritPrsrPutObjectToHandler(Handler, MS);
        IritPrsrFreeObject(MS);
    }
    IritPrsrCloseStream(Handler, TRUE);


    /** End **/
    IritMvarMVFree(DeformMV);
    IritUserMicroTileFree(MSRegularParam->Tile);

    for (i = 0; i < 3; ++i)
        IritFree(MSRegularParam->TilingSteps[i].TilesPerIntervals);

    /* Free the structure for the call back function. */
    IritTrivTVFree(TVMap);
}

int main(int argc, char** argv)
{
    GenerateMicroStructures();

    return 0;
}
