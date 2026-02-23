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

typedef CagdRType(*LclThicknessFuncCBType)(CagdRType u,
    CagdRType v,
    CagdRType w);

typedef struct UserMicroLocalDataStruct { /* User specific data in CB funcs. */
    TrivTVStruct* DefMap;
    CagdRType BndryThickness;
    LclThicknessFuncCBType ThicknessFuncCB;
} UserMicroLocalDataStruct;

static CagdRType UniformTilingCB(CagdRType u, CagdRType v, CagdRType w);
static CagdRType GradualThicknessCB(CagdRType u, CagdRType v, CagdRType w);
static CagdRType MidThinThicknessCB(CagdRType u, CagdRType v, CagdRType w);
static CagdRType MidThickThicknessCB(CagdRType u, CagdRType v, CagdRType w);
static CagdRType ThinningThirdCB(CagdRType u, CagdRType v, CagdRType w);
static CagdRType ThinningThird2CB(CagdRType u, CagdRType v, CagdRType w);

static int PreProcessTile1FaceParam(
    const UserMicroPreProcessTileCBStruct* CBData,
    TrivTVBndryType Bndry,
    CagdRType BndryThickness,
    LclThicknessFuncCBType ThicknessFuncCB,
    UserMicroTileBndryPrmStruct* BPrm);
static IPObjectStruct* PreProcessTile(IPObjectStruct* Tile,
    UserMicroPreProcessTileCBStruct* CBData);
static void GenerateMicroStructures(void);

/*****************************************************************************
* DESCRIPTION:								     *
*   A uniform tiler.							     *
*									     *
* PARAMETERS:								     *
*   u, v, w:	UVW location to	estimate thickness at, in the def. map.	     *
*									     *
* RETURN VALUE:								     *
*   CagdRType:								     *
*****************************************************************************/
static CagdRType UniformTilingCB(CagdRType u, CagdRType v, CagdRType w)
{
    // printf("%f   %f   %f\n", u, v, w);
    return 0.08;
}

/*****************************************************************************
* DESCRIPTION:								     *
*   A gradual thickness	changes	from the root (thick) to the tip (thin).     *
*									     *
* PARAMETERS:								     *
*   u, v, w:	UVW location to	estimate thickness at, in the def. map.	     *
*									     *
* RETURN VALUE:								     *
*   CagdRType:								     *
*****************************************************************************/
static CagdRType GradualThicknessCB(CagdRType u, CagdRType v, CagdRType w)
{
    return IRIT_BLEND(0.02, 0.10, w);	   /* w in [0, 1] -> [0.05, 0.25]. */
}

/*****************************************************************************
* DESCRIPTION:								     *
*   A gradual thickness	changes	from the root top (thick) to bottom (thin).  *
*									     *
* PARAMETERS:								     *
*   u, v, w:	UVW location to	estimate thickness at, in the def. map.	     *
*									     *
* RETURN VALUE:								     *
*   CagdRType:								     *
*****************************************************************************/
static CagdRType MidThickThicknessCB(CagdRType u, CagdRType v, CagdRType w)
{
    /* w in [0, 1] -> [0.02...0.145...0.02]. */
    return 0.02 + w * (1.0 - w) * 0.5;
}

/*****************************************************************************
* DESCRIPTION:								     *
*   A gradual thickness	changes	from the root top (thick) to bottom (thin).  *
*									     *
* PARAMETERS:								     *
*   u, v, w:	UVW location to	estimate thickness at, in the def. map.	     *
*									     *
* RETURN VALUE:								     *
*   CagdRType:								     *
*****************************************************************************/
static CagdRType MidThinThicknessCB(CagdRType u, CagdRType v, CagdRType w)
{
    /* w in [0, 1] -> [0.11...0.04...0.11]. */
    return 0.11 - w * (1.0 - w) * 0.357;
}

/*****************************************************************************
* DESCRIPTION:								     *
*   A thinning at 2/3 from the root.					     *
*									     *
* PARAMETERS:								     *
*   u, v, w:	UVW location to	estimate thickness at, in the def. map.	     *
*									     *
* RETURN VALUE:								     *
*   CagdRType:								     *
*****************************************************************************/
static CagdRType ThinningThirdCB(CagdRType u, CagdRType v, CagdRType w)
{
    w = 1.0 - w; /* w starts at the tip... */

    if (w < 0.25)
        return 0.1;
    else if (w > 0.4)
        return 0.1;
    else { /* In the thinning zone. */
        CagdRType t;

        if (w < 0.33) {/* 0.25 < w < 0.33. */
            t = (0.33 - w) / (0.33 - 0.21);

            return IRIT_BLEND(0.1, 0.02, t);
        }
        else { /* 0.4 > w > 0.33. */
            t = (w - 0.33) / (0.44 - 0.33);

            return IRIT_BLEND(0.1, 0.02, t);
        }
    }
}

/*****************************************************************************
* DESCRIPTION:								     *
*   A thinning at 2/3 from the root, at	the bottom of the MS.		     *
*									     *
* PARAMETERS:								     *
*   u, v, w:	UVW location to	estimate thickness at, in the def. map.	     *
*									     *
* RETURN VALUE:								     *
*   CagdRType:								     *
*****************************************************************************/
static CagdRType ThinningThird2CB(CagdRType u, CagdRType v, CagdRType w)
{
    CagdRType
        Thin1 = ThinningThirdCB(u, v, w);

    if (Thin1 >= 0.08)
        return Thin1;
    else
        return IRIT_BLEND(Thin1 * 0.5, 0.1, v);/* Make only bottom thinner. */
}

/*****************************************************************************
* DESCRIPTION:								     *
*   Prepare one	face parameters	for the	3D grid	tile synthesized on the	fly. *
*									     *
* PARAMETERS:								     *
*   LclMinDmn, LclMaxDmn:     UVW domain of this tile, in the parent	     *
*		deformation function.					     *
*   Bndry:	THe boundary (out of UMin/Max, VMin/Max, WMin/Max).	     *
*   BndryThickness:  To	set the	thickness of the synthesized boundary,	     *
*		or 0.0 to disable.					     *
*   ThicknessFuncCB: Call back function	to prescribe the thickness desired   *
*		based upon the UVW location.				     *
*   BPrm:	Parameters of this face	to update.			     *
*									     *
* RETURN VALUE:								     *
*   int:								     *
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
* DESCRIPTION:								     *
*									     *
*									     *
*									     *
* PARAMETERS:								     *
*   Tile:   Tile to preprocess.	 Here we expect	a NULL as we build tile	     *
*	    from scratch.						     *
*   CBData: The	call back data.						     *
*									     *
* RETURN VALUE:								     *
*   IPObjectStruct *:							     *
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

    fprintf(stderr, "Tile[%d,%d,%d] from (%.3f, %.3f %.3f) to (%.3f, %.3f, %.3f)\r",
        CBData->TileIdxs[0],
        CBData->TileIdxs[1],
        CBData->TileIdxs[2],
        LclMinDmn[0],
        LclMinDmn[1],
        LclMinDmn[2],
        LclMaxDmn[0],
        LclMaxDmn[1],
        LclMaxDmn[2]);

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

/*****************************************************************************
* DESCRIPTION:								     *
*   Create a micro structure with a varying-in-size tiling example.	     *
*									     *
* PARAMETERS:								     *
*   None								     *
*									     *
* RETURN VALUE:								     *
*   void								     *
*****************************************************************************/
static void GenerateMicroStructures(void)
{
    const char* InputDefMap = "Wing.itd"; // The wing
    int i, Handler;
    IPObjectStruct* MS, * DefMapPObj;
    MvarMVStruct* DeformMV;
    TrivTVStruct* TVMap;
    UserMicroParamStruct MSParam;
    UserMicroRegularParamStruct* MSRegularParam;
    UserMicroLocalDataStruct LclData;

    DefMapPObj = IritPrsrGetDataFiles(&InputDefMap, 1, FALSE, FALSE);
    if (DefMapPObj == NULL) {
        fprintf(stderr, "Failed to load the deformation function.\n");
        return;
    }
    assert(IP_IS_TRIVAR_OBJ(DefMapPObj));
    TVMap = DefMapPObj->U.Trivars;
    DeformMV = IritMvarCnvrtTVToMV(TVMap);

    /* Create the structure to be passed to the call back function. */
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

    MSRegularParam->TilingSteps[0].TilesPerIntervals[0] = 1;
    MSRegularParam->TilingSteps[1].TilesPerIntervals[0] = 1;
    MSRegularParam->TilingSteps[2].TilesPerIntervals[0] = 1;

    /* Call back function - will be called for each tile in the grid just   */
    /* before it is mapped through the deformation function, with the tile  */
    /* (that can be modified) and call back data.                           */
    MSRegularParam->PreProcessCBFunc = PreProcessTile;
    MSRegularParam->CBFuncData = &LclData;         /* The call back data. */

    /* 0. Uniform tiling. */
    fprintf(stderr, "\nMS: generating MS with uniform tiling along MS...\n");

    LclData.ThicknessFuncCB = UniformTilingCB;

    MS = IritUserMicroStructComposition(&MSParam); /* Construct microstructure. */

    Handler = IritPrsrOpenDataFile("MSUniform.itd", FALSE, 1);
    if (MS != NULL) {
        IritPrsrPutObjectToHandler(Handler, MS);
        IritPrsrFreeObject(MS);
    }
    IritPrsrCloseStream(Handler, TRUE);

    /* 1. Thinning at 2/3 along the MS from the root, at the top. */
    fprintf(stderr, "\nMS: generating MS with Thinning at 2/3 along the MS (top only)...\n");

    LclData.ThicknessFuncCB = ThinningThird2CB;

    MS = IritUserMicroStructComposition(&MSParam); /* Construct microstructure. */

    Handler = IritPrsrOpenDataFile("MSGradualThinningThirdBot.itd", FALSE, 1);
    if (MS != NULL) {
        IritPrsrPutObjectToHandler(Handler, MS);
        IritPrsrFreeObject(MS);
    }
    IritPrsrCloseStream(Handler, TRUE);

    fprintf(stderr, "\nMS: generating MS with Thinning at 2/3 along the MS...\n");

    LclData.ThicknessFuncCB = ThinningThirdCB;

    MS = IritUserMicroStructComposition(&MSParam); /* Construct microstructure. */

    Handler = IritPrsrOpenDataFile("MSGradualThinningThird2", FALSE, 1);
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
