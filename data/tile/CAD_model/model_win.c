/*****************************************************************************
*   Constructs locally varying trivariate tiles	in microstructure	     *
* constructions	using a	call back function.  The wing DefMap is	one example. *
******************************************************************************
* (C) Gershon Elber, Technion, Israel Institute	of Technology		     *
******************************************************************************
* Written by:  Gershon Elber				Ver 1.0, Dec 2017    *
*****************************************************************************/

#include "inc_irit/irit_sm.h"
#include "inc_irit/iritprsr.h"
#include "inc_irit/allocate.h"
#include "inc_irit/attribut.h"
#include "inc_irit/geom_lib.h"
#include "inc_irit/cagd_lib.h"
#include "inc_irit/user_lib.h"
#include "inc_irit/grap_lib.h"

typedef CagdRType(*LclThicknessFuncCBType)(CagdRType u,
    CagdRType v,
    CagdRType w);

typedef struct UserMicroLocalDataStruct { /* User specific data in CB funcs. */
    TrivTVStruct* DefMap;
    CagdRType BndryThickness;
    LclThicknessFuncCBType ThicknessFuncCB;
} UserMicroLocalDataStruct;

static CagdRType UniformTilingCB(CagdRType u, CagdRType v, CagdRType w);
static CagdRType* read_dims(int* ULength, int* VLength, int* WLength);

static int PreProcessTile1FaceParam(
    const UserMicroPreProcessTileCBStruct* CBData,
    TrivTVBndryType Bndry,
    CagdRType BndryThickness,
    LclThicknessFuncCBType ThicknessFuncCB,
    UserMicroTileBndryPrmStruct* BPrm);
static IPObjectStruct* PreProcessTile(IPObjectStruct* Tile,
    UserMicroPreProcessTileCBStruct* CBData);
static void GenerateMicroStructures(void);

static CagdRType UniformTilingCB(CagdRType u, CagdRType v, CagdRType w)
{
    int ULength, VLength, WLength;
    int UOrder = 2, VOrder = 2, WOrder = 2;

    CagdRType* ctrl = read_dims(&ULength, &VLength, &WLength);

    TrivTVStruct* TV = IritTrivBspTVNew(
        ULength, VLength, WLength,
        UOrder, VOrder, WOrder,
        CAGD_PT_E1_TYPE
    );

    int knotU = ULength + UOrder;  // total number of knots

    for (int i = 0; i < knotU; i++) {
        if (i < UOrder) {
            TV->UKnotVector[i] = 0.0;  // first k knots
        }
        else if (i >= ULength) {
            TV->UKnotVector[i] = 1.0;  // last k knots
        }
        else {
            TV->UKnotVector[i] = (double)(i - UOrder + 1) / (ULength - UOrder + 1);  // internal knots
        }
    }
    int knotV = VLength + VOrder;  // total number of knots

    for (int i = 0; i < knotV; i++) {
        if (i < VOrder) {
            TV->VKnotVector[i] = 0.0;  // first k knots
        }
        else if (i >= VLength) {
            TV->VKnotVector[i] = 1.0;  // last k knots
        }
        else {
            TV->VKnotVector[i] = (double)(i - VOrder + 1) / (VLength - VOrder + 1);  // internal knots
        }
    }
    int knotW = WLength + WOrder;  // total number of knots

    for (int i = 0; i < knotW; i++) {
        if (i < WOrder) {
            TV->WKnotVector[i] = 0.0;  // first k knots
        }
        else if (i >= WLength) {
            TV->WKnotVector[i] = 1.0;  // last k knots
        }
        else {
            TV->WKnotVector[i] = (double)(i - WOrder + 1) / (WLength - WOrder + 1);  // internal knots
        }
    }

    CagdRType UMin;
    CagdRType UMax;
    CagdRType VMin;
    CagdRType VMax;
    CagdRType WMin;
    CagdRType WMax;

    IritTrivTVDomain(TV, &UMin, &UMax, &VMin, &VMax, &WMin, &WMax);

    printf("TV domain: U [%f, %f], V [%f, %f], W [%f, %f]\n",
        UMin, UMax, VMin, VMax, WMin, WMax);

 

    //TrivTVStruct* TV = IritTrivBzrTVNew(
    //    ULength, VLength, WLength,
    //    CAGD_PT_E1_TYPE
    //);

    int n = VLength * ULength * WLength;
    printf("ULength: %d, VLength: %d, WLength: %d\n", ULength, VLength, WLength);


    printf("n: %d\n", n);
    for (int i = 0; i < ULength + UOrder; i++)
        printf("U knot[%d] = %f\n", i, TV->UKnotVector[i]);


    /* Copy all control points at once */
    memcpy(TV->Points[1], ctrl, sizeof(CagdRType) * n);



    /* Evaluate the trivariate */
    CagdRType* res = IritTrivTVEval2Malloc(TV, u, v, w);
    IritTrivTVFree(TV);
    IritFree(ctrl);

    return res[1];
}

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
    const UserMicroPreProcessTileCBStruct* CBData,
    TrivTVBndryType Bndry,
    CagdRType BndryThickness,
    LclThicknessFuncCBType ThicknessFuncCB,
    UserMicroTileBndryPrmStruct* BPrm)
{
    int i;
    CagdRType u, v, w;
    const CagdRType
        * LclMinDmn = CBData->TileLclDmnMin,
        * LclMaxDmn = CBData->TileLclDmnMax,
        * DefMapDmnMin = CBData->DefMapDmnMin,
        * DefMapDmnMax = CBData->DefMapDmnMax;

    IRIT_ZAP_MEM(BPrm, sizeof(UserMicroTileBndryPrmStruct));

    switch (Bndry) {
    case TRIV_U_MIN_BNDRY:
        u = LclMinDmn[0];
        v = (LclMinDmn[1] + LclMaxDmn[1]) * 0.5;
        w = (LclMinDmn[2] + LclMaxDmn[2]) * 0.5;
        break;
    case TRIV_U_MAX_BNDRY:
        u = LclMaxDmn[0];
        v = (LclMinDmn[1] + LclMaxDmn[1]) * 0.5;
        w = (LclMinDmn[2] + LclMaxDmn[2]) * 0.5;
        break;
    case TRIV_V_MIN_BNDRY:
        u = (LclMinDmn[0] + LclMaxDmn[0]) * 0.5;
        v = LclMinDmn[1];
        w = (LclMinDmn[2] + LclMaxDmn[2]) * 0.5;
        break;
    case TRIV_V_MAX_BNDRY:
        u = (LclMinDmn[0] + LclMaxDmn[0]) * 0.5;
        v = LclMaxDmn[1];
        w = (LclMinDmn[2] + LclMaxDmn[2]) * 0.5;
        break;
    case TRIV_W_MIN_BNDRY:
        u = (LclMinDmn[0] + LclMaxDmn[0]) * 0.5;
        v = (LclMinDmn[1] + LclMaxDmn[1]) * 0.5;
        w = LclMinDmn[2];
        break;
    case TRIV_W_MAX_BNDRY:
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
    case TRIV_U_MIN_BNDRY:
    case TRIV_U_MAX_BNDRY:
        BPrm->Bndry[0] = ThicknessFuncCB(u, LclMinDmn[1], LclMinDmn[2]);
        BPrm->Bndry[1] = ThicknessFuncCB(u, LclMaxDmn[1], LclMinDmn[2]);
        BPrm->Bndry[2] = ThicknessFuncCB(u, LclMinDmn[1], LclMaxDmn[2]);
        BPrm->Bndry[3] = ThicknessFuncCB(u, LclMaxDmn[1], LclMaxDmn[2]);
        break;
    case TRIV_V_MIN_BNDRY:
    case TRIV_V_MAX_BNDRY:
        BPrm->Bndry[0] = ThicknessFuncCB(LclMinDmn[0], v, LclMinDmn[2]);
        BPrm->Bndry[1] = ThicknessFuncCB(LclMaxDmn[0], v, LclMinDmn[2]);
        BPrm->Bndry[2] = ThicknessFuncCB(LclMinDmn[0], v, LclMaxDmn[2]);
        BPrm->Bndry[3] = ThicknessFuncCB(LclMaxDmn[0], v, LclMaxDmn[2]);
        break;
    case TRIV_W_MIN_BNDRY:
    case TRIV_W_MAX_BNDRY:
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
static IPObjectStruct* PreProcessTile(IPObjectStruct* Tile,
    UserMicroPreProcessTileCBStruct* CBData)
{
    UserMicroLocalDataStruct
        * LclData = (UserMicroLocalDataStruct*)CBData->CBFuncData;
    CagdRType UMin, UMax, VMin, VMax, WMin, WMax,
        * LclMinDmn = CBData->TileLclDmnMin,
        * LclMaxDmn = CBData->TileLclDmnMax,
        BndryThickness = LclData->BndryThickness;
    GMBBBboxStruct BBox;
    LclThicknessFuncCBType
        ThicknessFuncCB = LclData->ThicknessFuncCB;
    UserMicroTileBndryPrmStruct UMinPrms, UMaxPrms,
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

    if (!PreProcessTile1FaceParam(CBData, TRIV_U_MIN_BNDRY,
        0.0, ThicknessFuncCB, &UMinPrms) ||
        !PreProcessTile1FaceParam(CBData, TRIV_U_MAX_BNDRY,
            0.0, ThicknessFuncCB, &UMaxPrms) ||
        !PreProcessTile1FaceParam(CBData, TRIV_V_MIN_BNDRY,
            0.0, ThicknessFuncCB, &VMinPrms) ||
        !PreProcessTile1FaceParam(CBData, TRIV_V_MAX_BNDRY,
            0.0, ThicknessFuncCB, &VMaxPrms) ||
        !PreProcessTile1FaceParam(CBData, TRIV_W_MIN_BNDRY,
            0.0, ThicknessFuncCB, &WMinPrms) ||
        !PreProcessTile1FaceParam(CBData, TRIV_W_MAX_BNDRY,
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
        TrivTVStruct* TV;

        for (TV = Tile->U.Trivars; TV != NULL; TV = TV->Pnext) {
            MvarMVStruct
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
        IPObjectStruct* PTmp;
        TrivTVStruct* TV, * BTV,
            * BzrTVs = NULL;

        for (TV = Tile->U.Trivars; TV != NULL; TV = TV->Pnext) {
            if (TRIV_IS_BEZIER_TV(TV)) {
                BTV = IritTrivTVCopy(TV);
                IRIT_LIST_PUSH(BTV, BzrTVs);
            }
            else
                BzrTVs = IritCagdListAppend(IritTrivCnvrtBsp2BzrTV(TV), BzrTVs);
        }
        PTmp = IritPrsrLnkListToListObject(BzrTVs, IP_OBJ_TRIVAR);
        IritPrsrFreeObject(Tile);
        Tile = PTmp;
    }
#endif /* DEBUG_USER_MS_MAKE_TV_OBJS */

    return Tile;
}


static CagdRType* read_dims(int* ULength, int* VLength, int* WLength)
{
    IPObjectStruct* PObj;
    const char* DimsFileName = "dims.itd";

    PObj = IritPrsrGetDataFiles(&DimsFileName, 1, FALSE, FALSE);
    if (PObj == NULL) {
        fprintf(stderr, "Failed to load dims.itd\n");
        return NULL;
    }

    /* First value: ULength */
    if (PObj == NULL || !IP_IS_NUM_OBJ(PObj)) {
        fprintf(stderr, "First object is missing or not numeric\n");
        return NULL;
    }
    *ULength = (int)PObj->U.R;
    PObj = PObj->Pnext;

    /* Second value: VLength */
    if (PObj == NULL || !IP_IS_NUM_OBJ(PObj)) {
        fprintf(stderr, "Second object is missing or not numeric\n");
        return NULL;
    }
    *VLength = (int)PObj->U.R;
    PObj = PObj->Pnext;

    /* Third value: WLength */
    if (PObj == NULL || !IP_IS_NUM_OBJ(PObj)) {
        fprintf(stderr, "Third object is missing or not numeric\n");
        return NULL;
    }
    *WLength = (int)PObj->U.R;
    PObj = PObj->Pnext;

    int n = (*ULength) * (*VLength) * (*WLength);

    int i;
    CagdRType* ctrl = (CagdRType*)IritMalloc(sizeof(CagdRType) * n);
    IRIT_ZAP_MEM(ctrl, sizeof(CagdRType) * n);

    if (ctrl == NULL) {
        fprintf(stderr, "Allocation failed\n");
        return NULL;
    }

    for (i = 0; i < n; i++) {
        if (PObj == NULL || !IP_IS_NUM_OBJ(PObj)) {
            fprintf(stderr, "Invalid data at %d\n", i);
            IritFree(ctrl);
            return NULL;
        }

        ctrl[i] = PObj->U.R;
        PObj = PObj->Pnext;
    }

    return ctrl;
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
    IPObjectStruct* MS, * DefMapPObj;
    MvarMVStruct* DeformMV;
    TrivTVStruct* TVMap;
    UserMicroParamStruct MSParam;
    UserMicroRegularParamStruct* MSRegularParam;
    UserMicroLocalDataStruct LclData;
    printf("Loading deformation function...\n");

    DefMapPObj = IritPrsrGetDataFiles(&InputDefMap, 1, FALSE, FALSE);
    if (DefMapPObj == NULL) {
        fprintf(stderr, "Failed to load the deformation function.\n");
        return;
    }
    assert(IP_IS_TRIVAR_OBJ(DefMapPObj));
    TVMap = DefMapPObj->U.Trivars;
    DeformMV = IritMvarCnvrtTVToMV(TVMap);

    /* Create the structure to be passed to the callback function. */
    LclData.DefMap = TVMap;
    LclData.BndryThickness = 3;

    IRIT_ZAP_MEM(&MSParam, sizeof(UserMicroParamStruct));
    MSParam.TilingType = USER_MICRO_TILE_REGULAR;
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

    MS = IritUserMicroStructComposition(&MSParam);
    int tiles_num = MSRegularParam->TilingSteps[0].TilesPerIntervals[0] * MSRegularParam->TilingSteps[1].TilesPerIntervals[0] * MSRegularParam->TilingSteps[2].TilesPerIntervals[0];
    int params_per_tile = 7;
    CagdRType volume = 0;
    int index = 0;
    while (index < tiles_num) {
        IPObjectStruct* MV = MS->U.Lst.PObjList[index];
        int j = 0;
        while (j < params_per_tile) {
            IPObjectStruct* MQ = MV->U.Lst.PObjList[j];
            volume += fabs(IritTrivTVVolume(MQ->U.Trivars, TRUE));
            j++;
        }
        index++;
    }

    FILE* fp = fopen("props.txt", "w");
    if (fp == NULL) {
        perror("Failed to open file");
        return 1;
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
