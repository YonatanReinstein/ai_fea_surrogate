/******************************************************************************
* MergePly.c - merge points and	polylines into as-large-as-possible polylines.*
*******************************************************************************
* (C) Gershon Elber, Technion, Israel Institute	of Technology		      *
*******************************************************************************
* Written by Gershon Elber, Feb	2006.					      *
******************************************************************************/

#include <math.h>
#include <stdio.h>
#include "inc_irit/irit_sm.h"
#include "inc_irit/iritprsr.h"
#include "inc_irit/allocate.h"
#include "geom_loc.h"
#include "inc_irit/bool_lib.h"

#define GM_TEST_CLOSEST_DIST(V1, V2) { \
	IrtRType DstSqr; \
	if ((DstSqr = IRIT_PT_PT_DIST_SQR(V1 -> Coord, \
					  V2 -> Coord)) < MinSqr) { \
	    MinSqr = DstSqr; \
	} \
    }

#define GM_SET_LAST_VERTEX(Pl) \
			    (Pl) -> PAux = IritPrsrGetLastVrtx((Pl) -> PVertex)
#define GM_SET_LAST_VERTEX2(Pl, VLast) (Pl) -> PAux = VLast
#define GM_GET_LAST_VERTEX(Pl) ((IritPrsrVertexStruct *) (Pl) -> PAux)

typedef struct GMMergeEntitiesStruct {
    IrtRType MinDistSqr;      /* Minimal distance squared to another entity. */
    int MinDistSqrEntity;	/* Index of minimal distance squared entity. */
    int Valid;
    int Entity;					   /* This geometric entity. */
} GMMergeEntitiesStruct;

typedef struct GMMergeSortStruct {
    IrtRType Key;
    int EnttyIdx;
} GMMergeSortStruct;

typedef struct GMMergeSimilarPointsStruct {
    IrtRType *Pt;
    int Idx;
} GMMergeSimilarPointsStruct;

static void IritGeomMergeGeomInitPolys(VoidPtr GenericData, VoidPtr VPl);

static IrtRType IritGeomMergeGeomDistSqr2Polys(VoidPtr GenericData,
					       VoidPtr VPl1,
					       VoidPtr VPl2);
static IrtRType IritGeomMergeGeomKeyPoly1(VoidPtr GenericData, VoidPtr GE);
static IrtRType IritGeomMergeGeomKeyPoly2(VoidPtr GenericData, VoidPtr GE);
static int IritGeomMergeGeomMerge2Polys(VoidPtr GenericData,
				        void **VPl1,
				        void **VPl2);
static IrtRType GMMergeComputePointsDistSqr(int Dim,
					    GMMergeSimilarPointsStruct *PE1,
					    GMMergeSimilarPointsStruct *PE2);
#if defined(ultrix) && defined(mips)
static int GMMergeCompareEntities(void *GE1, void *GE2);
#else
static int GMMergeCompareEntities(const void *GE1, const void *GE2);
#endif /* ultrix && mips (no const support) */

/*****************************************************************************
* DESCRIPTION:								     *
*   Initialize the polylines to	be merged.  Here, it amounts to	caching	the  *
* last vertex for fast access.						     *
*									     *
* PARAMETERS:								     *
*   GenericData:  Optional data	to provide to merging call back	functions.   *
*   VPl:	  The polyline to initialize.				     *
*									     *
* RETURN VALUE:								     *
*   void								     *
*****************************************************************************/
static void IritGeomMergeGeomInitPolys(VoidPtr GenericData, VoidPtr VPl)
{
    IritPrsrPolygonStruct
	*Pl = (IritPrsrPolygonStruct *) VPl;

    GM_SET_LAST_VERTEX(Pl);
}

/*****************************************************************************
* DESCRIPTION:								     *
*   Computes minimal distance squared between end vertices of two polylines. *
*									     *
* PARAMETERS:								     *
*   GenericData:  Optional data	to provide to merging call back	functions.   *
*   VPl1, VPl2:	  To compute the minimal distance between the end points of. *
*									     *
* RETURN VALUE:								     *
*   IrtRType:	Minimal	distance squared computed.			     *
*****************************************************************************/
static IrtRType IritGeomMergeGeomDistSqr2Polys(VoidPtr GenericData, 
					       VoidPtr VPl1,
      VoidPtr VPl2)
{
    IrtRType
	MinSqr = IRIT_INFNTY;
    IritPrsrPolygonStruct
	*Pl1 = (IritPrsrPolygonStruct *) VPl1,
	*Pl2 = (IritPrsrPolygonStruct *) VPl2;
    IritPrsrVertexStruct
	*V1Start = Pl1 -> PVertex,
	*V1End = GM_GET_LAST_VERTEX(Pl1),
        *V2Start = Pl2 -> PVertex,
        *V2End = GM_GET_LAST_VERTEX(Pl2);

    GM_TEST_CLOSEST_DIST(V1Start, V2Start);
    GM_TEST_CLOSEST_DIST(V1End,   V2Start);
    GM_TEST_CLOSEST_DIST(V1Start, V2End);
    GM_TEST_CLOSEST_DIST(V1End,   V2End);

    return MinSqr;
}

/*****************************************************************************
* DESCRIPTION:								     *
*   Returns a key to sort the polylines	accordingly.			     *
*									     *
*									     *
* PARAMETERS:								     *
*   GenericData:  Optional data	to provide to merging call back	functions.   *
*   VPl:	Polyline to return its key (based on 1st point in polyline). *
*									     *
* RETURN VALUE:								     *
*   IrtRType:								     *
*****************************************************************************/
static IrtRType IritGeomMergeGeomKeyPoly1(VoidPtr GenericData, VoidPtr VPl)
{
    IritPrsrPolygonStruct
        *Pl = (IritPrsrPolygonStruct *) VPl;
    IrtRType
	*Coord = Pl -> PVertex -> Coord;

    /* Use the XY coordinates of the first vertex as the key. */
    return Coord[0] * 3 + Coord[1] * 7;
}

/*****************************************************************************
* DESCRIPTION:								     *
*   Returns a key to sort the polylines	accordingly.			     *
*									     *
*									     *
* PARAMETERS:								     *
*   GenericData:  Optional data	to provide to merging call back	functions.   *
*   VPl:       Polyline	to return its key (based on last point in polyline). *
*									     *
* RETURN VALUE:								     *
*   IrtRType:								     *
*****************************************************************************/
static IrtRType IritGeomMergeGeomKeyPoly2(VoidPtr GenericData, VoidPtr VPl)
{
    IritPrsrPolygonStruct
        *Pl = (IritPrsrPolygonStruct *) VPl;
    IrtRType
        *Coord = GM_GET_LAST_VERTEX(Pl) -> Coord;

    /* Use the XY coordinates of the first vertex as the key. */
    return Coord[0] * 3 + Coord[1] * 7;
}

/*****************************************************************************
* DESCRIPTION:								     *
*   Merges two polylines into one.  The	address	that references	Pl2 is set   *
* to NULL after	Pl2 is freed.  Pl1 will	hold the merged	poly.		     *
*									     *
* PARAMETERS:								     *
*   GenericData:   Optional data to provide to merging call back functions.  *
*   VPl1, VPl2:	   The two polylines to	merge into one.			     *
*									     *
* RETURN VALUE:								     *
*   int:     TRUE if successful, FALSE otherwise.			     *
*****************************************************************************/
static int IritGeomMergeGeomMerge2Polys(VoidPtr GenericData, 
				        void **VPl1,
				        void **VPl2)
{
    IritPrsrPolygonStruct
	**Pl1 = (IritPrsrPolygonStruct **) VPl1,
	**Pl2 = (IritPrsrPolygonStruct **) VPl2;
    IritPrsrVertexStruct
	*V1Start = (*Pl1) -> PVertex,
	*V1End = GM_GET_LAST_VERTEX(*Pl1),
        *V2Start = (*Pl2) -> PVertex,
        *V2End = GM_GET_LAST_VERTEX(*Pl2);
    IrtRType
	d00 = IRIT_PT_PT_DIST_SQR(V1Start -> Coord, V2Start -> Coord),
	d10 = IRIT_PT_PT_DIST_SQR(V1End -> Coord,   V2Start -> Coord),
	d01 = IRIT_PT_PT_DIST_SQR(V1Start -> Coord, V2End -> Coord),
	d11 = IRIT_PT_PT_DIST_SQR(V1End -> Coord,   V2End -> Coord);

    if (d00 < d10 && d00 < d01 && d00 < d11) {/* Merge Start Pl1 & Start Pl2.*/
        (*Pl1) -> PVertex = V1Start =
	    IritPrsrReverseVrtxList2(V1End = V1Start);
    }
    else if (d10 < d01 && d10 < d11) {	       /* Merge End Pl1 & Start Pl2. */
    }
    else if (d01 < d11) {		       /* Merge Start Pl1 & End Pl2. */
        IRIT_SWAP(IritPrsrVertexStruct *, (*Pl1) -> PVertex, (*Pl2) -> PVertex);
	IRIT_SWAP(IritPrsrVertexStruct *, V1Start, V2Start);
	IRIT_SWAP(IritPrsrVertexStruct *, V1End, V2End);
    }
    else {				         /* Merge End Pl1 & End Pl2. */
        (*Pl2) -> PVertex = V2Start =
	    IritPrsrReverseVrtxList2(V2End = V2Start);
    }

    /* Is merged point identical?  If so purge one of the two point. */
    if (IRIT_PT_APX_EQ_EPS(V1End -> Coord, V2Start -> Coord, IRIT_EPS)) {
	V1End -> Pnext = V2Start -> Pnext; 
	V2Start -> Pnext = NULL;
	if (V2Start -> Attr != NULL)
	    IRIT_MISC_ATTR_SAFECOPY_ATTRS(V1End -> Attr, V2Start -> Attr);
	if (IRIT_PRSR_HAS_NORMAL_VRTX(V2Start)) {
	    IRIT_VEC_COPY(V1End -> Normal, V2Start -> Normal);
	    IRIT_PRSR_SET_NORMAL_VRTX(V1End);
	}
    }
    else {
	V1End -> Pnext = V2Start;
	(*Pl2) -> PVertex = NULL;
    }
    IritPrsrFreePolygon(*Pl2);
    *Pl2 = NULL;

    GM_SET_LAST_VERTEX2(*Pl1, V2End);

    return TRUE;
}

/*****************************************************************************
* DESCRIPTION:								     *
*   Routine to compare two geometric entities' minimal distance	for sorting  *
* purposes.								     *
*									     *
* PARAMETERS:								     *
*   GE1, GE2:  Two pointers to GMMergeEntitiesStruct structs.		     *
*									     *
* RETURN VALUE:								     *
*   int:   >0, 0, or <0	as the relation	between	the two	distances.	     *
*****************************************************************************/
#if defined(ultrix) && defined(mips)
static int GMMergeCompareEntities(void *GE1, void *GE2)
#else
static int GMMergeCompareEntities(const void *GE1, const void *GE2)
#endif /* ultrix && mips (no const support) */
{
    IrtRType
        Diff = (*((GMMergeSortStruct **) GE1)) -> Key -
	       (*((GMMergeSortStruct **) GE2)) -> Key;

    return IRIT_SIGN(Diff);
}

/*****************************************************************************
* DESCRIPTION:								     M
*   Merges separated geometric entities	into longer ones, in place, as much  M
* as possible. Given a list of entities	(typically points or polylines),     M
* find and merge closest ones as possible and in place.	 This function	     M
* should only be used to merge geometry	that is	very precise and adjacent    M
* entities distance is very small.					     M
*									     *
* PARAMETERS:								     M
*   GeomEntities:    Geometry to merge,	typically points or polylines, as a  M
*		     vector of reference pointers to the entities.	     M
*		       Returned, merged, data will also	be stored here.	     M
*   NumOfGEntities:  Number of geometric entities.  Also length	of	     M
*		     GeomEntities.					     M
*   IdenticalEps:    Epsilon to	consider two entities' distance	as zero.     M
*		     This epsilon is typically very small, i.e., IRIT_UEPS.  M
*   GenericData:     Optional (can be NULL) data to transfer to	all call     M
*		     back functions.					     M
*   InitFunc:	     A function	to initialize all geometry.  NULL for the    M
*		     initialization of IritPrsrPolygonStruct polylines.	     M
*   DistSqrFunc:     A distance	computation function. NULL for two	     M
*		     IritPrsrPolygonStruct polylines compare. This function  M
*		     computes the minimal distance squared between two	     M
*		     entities.						     M
*   KeyFuncs:	     Two function to return two	Keys to	sort the entities so M
*		     that similar/adjacent entities will get similar a Key.  M
*		       For example for a polyline, they	will return a key    M
*		     related to	the two	end points of the polyline.	     M
*   MergeFunc:	     A merge function. NULL to merge two		     M
*		     IritPrsrPolygonStructs polylines merge.		     M
*		       This function destroys the two input		     M
*		     entities and returns the merged entity in the first     M
*		     reference,	setting	the second reference to	NULL.	     M
*									     *
* RETURN VALUE:								     M
*   int:	     Number of merged entities in the end or 0 if error.     M
*									     *
* SEE ALSO:								     M
*   IritGeomMergeGeometry						     M
*									     *
* KEYWORDS:								     M
*   IritGeomMergeSameGeometry						     M
*****************************************************************************/
int IritGeomMergeSameGeometry(void **GeomEntities,
			      int NumOfGEntities,
			      IrtRType IdenticalEps,
			      VoidPtr GenericData,
			      IritGeomMergeGeomInitFuncType InitFunc,
			      IritGeomMergeGeomDistFuncType DistSqrFunc,
			      IritGeomMergeGeomKeyFuncType *KeyFuncs,
			      IritGeomMergeGeomMergeFuncType MergeFunc)
{
    int i, j, k, NKeys, NKeyFuncs,
	N = NumOfGEntities;
    IrtRType
        IdenticalEpsSqr = IRIT_SQR(IdenticalEps);
    IritGeomMergeGeomKeyFuncType LclKeyFuncs[3];
    GMMergeSortStruct *KeysData, **KeysRefs;

    if (NumOfGEntities <= 1)
        return NumOfGEntities;

    if (InitFunc == NULL)
        InitFunc = IritGeomMergeGeomInitPolys;
    if (DistSqrFunc == NULL)
        DistSqrFunc = IritGeomMergeGeomDistSqr2Polys;
    if (KeyFuncs == NULL) {
	LclKeyFuncs[0] = IritGeomMergeGeomKeyPoly1;
	LclKeyFuncs[1] = IritGeomMergeGeomKeyPoly2;
	LclKeyFuncs[2] = NULL;
	KeyFuncs = LclKeyFuncs;
    }
    if (MergeFunc == NULL)
        MergeFunc = IritGeomMergeGeomMerge2Polys;

    /* Initialize all geometry. */
    for (i = 0; i < N; i++)
        InitFunc(GenericData, GeomEntities[i]);

    /* Sort the data to localize the entities. */
    for (NKeyFuncs = 0; KeyFuncs[NKeyFuncs] != NULL; NKeyFuncs++)
        ;
    assert(NKeyFuncs <= 2 && NKeyFuncs > 0); /* Support up to two key funcs. */

    NKeys = N * NKeyFuncs;
    KeysData = (GMMergeSortStruct *) IritMalloc(sizeof(GMMergeSortStruct)
						                    * NKeys);
    KeysRefs = (GMMergeSortStruct **) IritMalloc(sizeof(GMMergeSortStruct *)
						                    * NKeys);
    for (i = k = 0; i < N; i++) {
        for (j = 0; j < NKeyFuncs; j++, k++) {
	    KeysData[k].Key = KeyFuncs[j](GenericData, GeomEntities[i]);
	    KeysData[k].EnttyIdx = i;
	}	    
    }

    for (i = 0; i < NKeys; i++)
        KeysRefs[i] = &KeysData[i];

    qsort(KeysRefs, NKeys, sizeof(GMMergeSortStruct *),
	  GMMergeCompareEntities);

    /* Merge adjacent entities (that must have same keys). */
    for (i = 0; i < NKeys; i++) {
        int IRef = KeysRefs[i] -> EnttyIdx;

	if (IRef < 0)
	    continue;

        for (j = i + 1; j < NKeys; j++) {              /* Do not check self. */
	    int JRef = KeysRefs[j] -> EnttyIdx;
	    IrtRType OtherKeyVal;

	    if (JRef < 0 ||
		JRef == IRef ||
		GeomEntities[IRef] == GeomEntities[JRef])
	        continue;

	    /* Examine only IdenticalEps near entities. */
	    if (KeysRefs[j] -> Key - KeysRefs[i] -> Key > IdenticalEpsSqr)
	        break;

	    if (DistSqrFunc(GenericData, GeomEntities[IRef],
			    GeomEntities[JRef]) < IdenticalEpsSqr &&
		MergeFunc(GenericData, &GeomEntities[IRef],
			  &GeomEntities[JRef])) {
	        assert(GeomEntities[JRef] == NULL &&
		       GeomEntities[IRef] != NULL);

		/* Here we assume exactly two key functions per entry.      */
		/* Disable other entry of Entity[j] (adjacent in KeysData): */
		if (IRIT_APX_EQ_EPS(KeysData[JRef * 2].Key, KeysRefs[j] -> Key,
				    IdenticalEps))
		    OtherKeyVal = KeysData[JRef * 2 + 1].Key;
		else
		    OtherKeyVal = KeysData[JRef * 2].Key;

		KeysData[JRef * 2].EnttyIdx = -1;               /* Invalid. */
		KeysData[JRef * 2 + 1].EnttyIdx = -1;           /* Invalid. */

		/* Update Entty[i] (now the merged entity) this end key on  */
		/* Entty[j] other end key.				    */
		if (IRIT_APX_EQ_EPS(KeysData[IRef * 2].Key, KeysRefs[i] -> Key,
				    IdenticalEps))
		    KeysData[IRef * 2].Key = OtherKeyVal;
		else
		    KeysData[IRef * 2 + 1].Key = OtherKeyVal;
	    }
	}
    }

    /* Clean the vector from NULL entities. */
    for (j = N - 1; GeomEntities[j] == NULL; N = j--)
        ;
    for (i = 0; i < j; i++) {
        if (GeomEntities[i] == NULL) {
	    GeomEntities[i] = GeomEntities[N = j--];
	    for ( ; GeomEntities[j] == NULL; N = j--);
	}
    }

    IritFree(KeysData);
    IritFree(KeysRefs);

    return N;
}

/*****************************************************************************
* DESCRIPTION:								     M
*   Merges separated geometric entities	into longer ones, in place, as much  M
* as possible. Given a list of entities	(typically points or polylines),     M
* find and merge closest ones as possible and in place.			     M
*									     *
* PARAMETERS:								     M
*   GeomEntities:    Geometry to merge,	typically points or polylines, as a  M
*		     vector of reference pointers to the entities.	     M
*		       Returned, merged, data will also	be stored here.	     M
*   NumOfGEntities:  Number of geometric entities.  Also length	of	     M
*		     GeomEntities.					     M
*   Eps:	     Epsilon of	similarity to merge entities at.  Entities   M
*		     farther than Eps will not be merged.		     M
*   IdenticalEps:    Epsilon to	consider two entities' distance	as zero.     M
*   GenericData:     Optional (can be NULL) data to transfer to	all call     M
*		     back functions.					     M
*   InitFunc:	     A function	to initialize all geometry.  NULL for the    M
*		     initialization of IritPrsrPolygonStruct polylines.	     M
*   DistSqrFunc:     A distance	computation function. NULL for two	     M
*		     IritPrsrPolygonStruct polylines compare.  This function M
*		     computes the minimal distance squared between two	     M
*		     entities.						     M
*   KeyFuncs:	     A NULL terminated version of functions to return a	Key  M
*		     to	sort the entities so that similar/adjacent entities  M
*		     will get a	similar	Key.				     M
*		       A polyline, for example,	can have two Key functions   M
*		     of	its two	end points (to be merged).		     M
*   MergeFunc:	     A merge function. NULL to merge two		     M
*		     IritPrsrPolygonStructs polylines merge.		     M
*		     This function returns a merged entity		     M
*		     while destroying the two input entities.		     M
*									     *
* RETURN VALUE:								     M
*   int:	     Number of merged entities in the end or 0 if error.     M
*									     *
* SEE ALSO:								     M
*   IritMvarPolyMergePolylines,	IritGeomMergePolylines,			     M
*   IritGeomMergeSameGeometry						     M
*									     *
* KEYWORDS:								     M
*   IritGeomMergeGeometry, merge, points, polylines			     M
*****************************************************************************/
int IritGeomMergeGeometry(void **GeomEntities,
			  int NumOfGEntities,
			  IrtRType Eps,
			  IrtRType IdenticalEps,
			  VoidPtr GenericData,
			  IritGeomMergeGeomInitFuncType InitFunc,
			  IritGeomMergeGeomDistFuncType DistSqrFunc,
			  IritGeomMergeGeomKeyFuncType *KeyFuncs,
			  IritGeomMergeGeomMergeFuncType MergeFunc)
{
    int i, j, N;
    IrtRType
	EpsSqr = IRIT_SQR(Eps),
        IdenticalEpsSqr = IRIT_SQR(IdenticalEps);
    GMMergeEntitiesStruct *GE;
    IritGeomMergeGeomKeyFuncType LclKeyFuncs[3];

    if (NumOfGEntities <= 1)
        return NumOfGEntities;

    if (InitFunc == NULL)
        InitFunc = IritGeomMergeGeomInitPolys;
    if (DistSqrFunc == NULL)
        DistSqrFunc = IritGeomMergeGeomDistSqr2Polys;
    if (KeyFuncs == NULL) {
	LclKeyFuncs[0] = IritGeomMergeGeomKeyPoly1;
	LclKeyFuncs[1] = IritGeomMergeGeomKeyPoly2;
	LclKeyFuncs[2] = NULL;
	KeyFuncs = LclKeyFuncs;
    }
    if (MergeFunc == NULL)
        MergeFunc = IritGeomMergeGeomMerge2Polys;

    /* Merge, as possible, segments that are close to within IdenticalEps. */
    do {
        NumOfGEntities = IritGeomMergeSameGeometry(GeomEntities,
						   i = NumOfGEntities,
						   IdenticalEps, GenericData,
						   InitFunc, DistSqrFunc,
						   KeyFuncs, MergeFunc);
    }
    while (i > NumOfGEntities);

    if (NumOfGEntities <= 1)
        return NumOfGEntities;

    GE = (GMMergeEntitiesStruct *)
	IritMalloc(sizeof(GMMergeEntitiesStruct) * NumOfGEntities);

    /* Initialize all geometry. */
    N = NumOfGEntities;
    for (i = 0; i < N; i++)
        InitFunc(GenericData, GeomEntities[i]);

    /* Entering main iteration loop: */
    do {
        NumOfGEntities = N;

        /* Initialize the data structure. */
        for (i = 0; i < N; i++) {
	    GE[i].Entity = i;
	    GE[i].MinDistSqr = IRIT_INFNTY;
	    GE[i].MinDistSqrEntity = -1;
	    GE[i].Valid = TRUE;
	}

        /* Compute minima distance between entity i and all other entities. */
        for (i = 0; i < N; i++) {
	    for (j = i + 1; j < N; j++) {
	        IrtRType
		    Dst = DistSqrFunc(GenericData,
				      GeomEntities[i], GeomEntities[j]);

		if (GE[i].MinDistSqr > Dst) {
		    GE[i].MinDistSqr = Dst;
		    GE[i].MinDistSqrEntity = j;
		}
		if (GE[j].MinDistSqr > Dst) {
		    GE[j].MinDistSqr = Dst;
		    GE[j].MinDistSqrEntity = i;
		}
	    }
	}

	/* Merge pairs of closest entities. A merged entity will not be     */
	/* remerged in the same cycle.					    */
	for (i = 0; i < N; i++) {
	    /* Examine the validity of this pair. Further, also examine if  */
	    /* the distance is virtually zero (common in chaining precise   */
	    /* data sets) or they both see each other as their closest.     */
	    j = GE[i].MinDistSqrEntity;
	    if (j >= 0 &&
		GE[i].Valid &&
		GE[j].Valid &&
		(GE[i].MinDistSqr < IdenticalEpsSqr ||
		 (GE[j].MinDistSqrEntity == i &&
		  GE[i].MinDistSqr <= EpsSqr))) {
	        MergeFunc(GenericData,
			  &GeomEntities[GE[i].Entity],
			  &GeomEntities[GE[j].Entity]);
		GE[i].Valid = GE[j].Valid = FALSE;
	    }
	}

	/* Clean the vector from NULL entities. */
	for (j = N - 1; GeomEntities[j] == NULL; N = j--);
	for (i = 0; i < j; i++) {
	    if (GeomEntities[i] == NULL) {
		GeomEntities[i] = GeomEntities[N = j--];
		for ( ; GeomEntities[j] == NULL; N = j--);
	    }
	}

#       ifdef DEBUG
	{
	    IRIT_SET_IF_DEBUG_ON_PARAMETER(_DebugMergeGeometry, FALSE) {
	        IRIT_INFO_MSG_PRINTF("%d input entites, merged into %d entities\n",
				     NumOfGEntities, N);
	    }
	}
#       endif /* DEBUG */
    }
    while (NumOfGEntities != N);

#   ifdef DEBUG
    {
        IRIT_SET_IF_DEBUG_ON_PARAMETER(_DebugMergeGeometry, FALSE) {
	    IRIT_INFO_MSG_PRINTF("%d input entities, merged into %d entities\n",
				 NumOfGEntities, N);
	    for (i = 0; i < N; i++) {
	        IritPrsrPolygonStruct
		    *Pl = (IritPrsrPolygonStruct *) GeomEntities[i];
		IritPrsrVertexStruct
		    *V1 = Pl -> PVertex,
		    *V2 = IritPrsrGetLastVrtx(V1);

		IRIT_INFO_MSG_PRINTF(
			"Poly %d [%.8lg %.8lg %.8lg] [%.8lg %.8lg %.8lg]\n", i,
			V1 -> Coord[0], V1 -> Coord[1], V1 -> Coord[2],
			V2 -> Coord[0], V2 -> Coord[1], V2 -> Coord[2]);
	    }
	}
    }
#   endif /* DEBUG */

    IritFree(GE);

    return N;
}

/*****************************************************************************
* DESCRIPTION:								     M
*   This routines combines a given polygon with	its interior closed loops    M
* by adding an edge from the polygon boundary to each of its closed loops    M
*   For	example:							     M
*									     V
* +-----------------------+      +-----------------------+		     V
* |                       |	 |                       |		     V
* |  / \         / \      |	 |  / \________ / \      |		     V
* |  \ /        |   |     |	 |  \ /        |   |_____|		     V
* |      _       \ /      | -->  |      _       \ /      |		     V
* |     / \_              | -->  |     / \_              |		     V
* |    |    |             |	 |    |    |_____________|		     V
* |     \__/              |	 |     \__/              |		     V
* |                       |      |                       |		     V
* +-----------------------+      +-----------------------+		     V
*									     V
*   Algorithm:								     M
* 1. Transform the polygon and the closed loops to a plane parallel to XY    V
*    plane (Z = Const plane).						     V
* 2. For each loop find its MaxX while keeping the information on the        V
*    vertex on that extremum.						     V
* 3. For the loop with the biggest MaxX:				     V
*    3.1. based on the sign of the area decide to reverse (or not) the inner V
*         loop.								     V
*    3.2. Fire a ray from the extremum vertex, to the +X direction outside   V
*         of the loop till it intersect the polygon, break the polygon at    V
*         that point and add two edges to beginning of loop from breaking    V
*         point and from end of loop to breaking point (beginning/end point  V
*	  of loop is the extrema vertex point).				     V
* 4. Repeat step 3, with all loops.					     V
* 5. Transform the new polygon back (using the inverse matrix of step 1)     V
*    to its original orientation.					     V
*									     *
* PARAMETERS:								     M
*   PlMain:   Main polygon to combine with the given closed loops, in place. M
*   PClosedPls:	Closed loops inside polygon PlMain, to merge with, in place. M
*									     *
* RETURN VALUE:								     M
*   IritPrsrPolygonStruct *:  Merged polygon, based on PlMain, or NULL	     M
*                             if error.					     M
*									     *
* SEE ALSO:								     M
*   IritMvarPolyMergePolylines,	IritGeomMergePolylines,			     M
*   IritGeomMergeSameGeometry						     M
*									     *
* KEYWORDS:								     M
*   IritGeomMergeClosedLoopHoles, merge, polygons, holes		     M
*****************************************************************************/
IritPrsrPolygonStruct *IritGeomMergeClosedLoopHoles(
					    IritPrsrPolygonStruct *PlMain,
					    IritPrsrPolygonStruct *PClosedPls)
{
    IrtRType MaxX, PlMainArea, Area,
        ZAxis[3] = { 0.0, 0.0, 1.0 };
    IrtPlnType Pln;
    IritPrsrVertexStruct *V, *VMaxX, *Vnext, *VHead;
    IrtHmgnMatType RotMat, InvRotMat;
    IritPrsrPolygonStruct *Pl;

    IRIT_PLANE_COPY(Pln, PlMain -> Plane);

    /* Stage 1 - Transform the vertices to a plane parallel to XY plane: */
    IritGeomGenRotate2ZMatrix(RotMat, PlMain -> Plane);
    V = VHead = PlMain -> PVertex;
    do {				         /* Transform the vertices. */
        IritMiscMatMultPtby4by4(V -> Coord, V -> Coord, RotMat);
	V = V -> Pnext;
    }
    while (V != NULL && V != VHead);
    IRIT_PT_COPY(PlMain -> Plane, ZAxis);
    PlMain -> Plane[3] = -PlMain -> PVertex -> Coord[2];

    Pl = PClosedPls;
    while (Pl != NULL) {   		   /* And transform the loops also. */
        V = VHead = Pl -> PVertex;
	do {			                 /* Transform the vertices. */
	    IritMiscMatMultPtby4by4(V -> Coord, V -> Coord, RotMat);
	    V = V -> Pnext;
	}
	while (V != NULL && V != VHead);

	IRIT_PT_COPY(Pl -> Plane, ZAxis);
	Pl -> Plane[3] = -Pl -> PVertex -> Coord[2];

	Pl = Pl -> Pnext;
    }

    IritMiscMatTranspMatrix(RotMat, InvRotMat);     /* Compute the inverse. */

    PlMainArea = IritGeomPolyOnePolyArea(PlMain, FALSE);
    assert(PlMainArea > IRIT_EPS);

    /* Stage 2 - find MaxX extremum value of all given loops. */
    Pl = PClosedPls;
    while (Pl != NULL) {  		   /* And transform the loops also. */
        V = VHead = Pl -> PVertex;
	MaxX = V -> Coord[0];
	VMaxX = V;
	do {			                 /* Transform the vertices. */
	    if (MaxX < V -> Coord[0]) {
		MaxX = V -> Coord[0];
		VMaxX = V;
	    }
	    V = V -> Pnext;
	}
	while (V != NULL && V != VHead);

	IritMiscAttrIDSetRealAttrib(&Pl -> Attr,
				    IRIT_ATTR_CREATE_ID(_MaxX), MaxX);
	IritMiscAttrIDSetRefPtrAttrib(&Pl -> Attr,
				      IRIT_ATTR_CREATE_ID(_VMaxX), VMaxX);

	Pl = Pl -> Pnext;
    }

    /* Make PlMain vertex list circular. */
    IritPrsrGetLastVrtx(PlMain -> PVertex) -> Pnext = PlMain -> PVertex;

    /* Stage 3/4 - find the loop with biggest MaxX and combine it with the  */
    /* polygon itself. Do it for all closed loops, in list:		    */
    while (PClosedPls != NULL) {
      IritPrsrPolygonStruct *MaxXPl;
	IrtRType ThisMaxX;

	MaxX = -IRIT_INFNTY;
	MaxXPl = NULL;

	/* Find the loop with maximum MaxX, and remove it from PClosed list. */
	Pl = PClosedPls;
	while (Pl != NULL) {   		   /* And transform the loops also. */
	    ThisMaxX = IritMiscAttrIDGetRealAttrib(Pl -> Attr,
						   IRIT_ATTR_CREATE_ID(_MaxX));
	    
	    if (MaxX < ThisMaxX) {
	        MaxX = ThisMaxX;
		MaxXPl = Pl;
	    }

	    Pl = Pl -> Pnext;
	}

	assert(MaxXPl != NULL);

	if (MaxXPl == PClosedPls) {
	    PClosedPls = PClosedPls -> Pnext;
	}
	else {
	    IritPrsrPolygonStruct
	        *PrevPl = IritPrsrGetPrevPoly(PClosedPls, MaxXPl);

	    assert(PrevPl -> Pnext == MaxXPl);
	    PrevPl -> Pnext = MaxXPl -> Pnext;
	}
	MaxXPl -> Pnext = NULL;
	VMaxX = IritMiscAttrIDGetRefPtrAttrib(MaxXPl -> Attr,
					      IRIT_ATTR_CREATE_ID(_VMaxX));

	/* Stage 3.1. Now test if the vertex list should be reversed. Areas */
	/* of PlMain and of MaxXPl should be reversed.			    */
	Area = IritGeomPolyOnePolyArea(MaxXPl, FALSE);
	assert(Area > IRIT_EPS);
	if (PlMainArea * Area > 0) {
	    /* Make MaxXPl vertex list circular. */
	    IritPrsrGetLastVrtx(MaxXPl -> PVertex) -> Pnext = MaxXPl -> PVertex;
	    IritPrsrReverseVrtxList(MaxXPl);
	    V = VHead = MaxXPl -> PVertex;
	    do {		                      /* Reset the normals. */
	        IRIT_PRSR_RST_NORMAL_VRTX(V);
	        V = V -> Pnext;
	    }
	    while (V != VHead);
	}

	/* Stage 3.2. O.k. lets fire a ray from VMaxX to +X direction and   */
	/* see where it intersects the polygon. The routine 		    */
	/* IritBoolCutPolygonAtRay will add two vertices at the ray         */
	/* intersection into polygon vertex list and return a pointer to    */
	/* first one, so we can chain our loop directly between these two   */
	/* new vertices.						    */
	V = IritBoolCutPolygonAtRay(PlMain, VMaxX -> Coord);
	Vnext = V -> Pnext;

	/* Introduce a copy of VMaxX and successor to VMaxX: */
	VMaxX -> Pnext = IritPrsrAllocVertex(VMaxX -> Tags, NULL,
					     VMaxX -> Pnext);
	IRIT_PT_COPY(VMaxX -> Pnext -> Coord, VMaxX -> Coord);
	IRIT_VEC_RESET(VMaxX -> Pnext -> Normal);
	V -> Pnext = VMaxX -> Pnext;
	IRIT_PRSR_SET_INTERNAL_VRTX(V);
	VMaxX -> Pnext = Vnext;
	IRIT_PRSR_SET_INTERNAL_VRTX(VMaxX);

	MaxXPl -> PVertex = NULL;
	IritPrsrFreePolygon(MaxXPl);
    }

    /* Stage 5 - Time to rotate polygon back to its original position. */
    V = VHead = PlMain -> PVertex;
    do {
	IritMiscMatMultPtby4by4(V -> Coord, V -> Coord, InvRotMat);
	V = V -> Pnext;
    }
    while (V != NULL && V != VHead);

    IRIT_PRSR_RST_CONVEX_POLY(PlMain);/* This polygon is not convex any more.*/
    IRIT_PLANE_COPY(PlMain -> Plane, Pln);

    return PlMain;
}

/*****************************************************************************
* DESCRIPTION:								     *
*   Routine to compute the distance square between two point entities.	     *
*									     *
* PARAMETERS:								     *
*   Dim:       Dimension of points.					     *
*   PE1, PE2:  Two pointers to GMMergeSimilarPointsStruct structs.	     *
*									     *
* RETURN VALUE:								     *
*   int:   >0, 0, or <0	as the relation	between	the two	distances.	     *
*****************************************************************************/
static IrtRType GMMergeComputePointsDistSqr(int Dim,
					    GMMergeSimilarPointsStruct *PE1,
					    GMMergeSimilarPointsStruct *PE2)
{
    int i;
    CagdRType
	d = 0;

    for (i = 0; i < Dim; i++)
        d += IRIT_SQR(PE1 -> Pt[i] - PE2 -> Pt[i]);

    return d;
}

/*****************************************************************************
* DESCRIPTION:								     M
*    Finds out similar (to Tolerance) points in	the given vector of points   M
* of length VecLen.  Returned is a newly allocated vector VecIdx of length   M
* VecLen so VecIdx[i] =	j means	point i	is the same as point j.		     M
*									     *
* PARAMETERS:								     M
*   VecPts:	  Vector of points to seek similarities.		     M
*   PtLen:	  Number of real valued	coordinates in point (2	for E2,	4    M
*		  for P3, etc.).					     M
*   VecLen:	  Length of VecPts. VecPts should have PtLen * VecLen reals. M
*   Tolerance:	  To consider two points the same, in L^2 norm.		     M
*									     *
* RETURN VALUE:								     M
*   int	*:	  The dynamically allocated vector of indices, so that	     M
*		  VecIdx[i] = j	 means i and j are the same.		     M
*									     *
* SEE ALSO:								     M
*   IritGeomMergeSameGeometry						     M
*									     *
* KEYWORDS:								     M
*   IritGeomMergeFindSimilarPoints, merge				     M
*****************************************************************************/
static unsigned int GMMergeHashCell(const int *Cell, int Dim)
{
    unsigned int
        h = 2166136261u;
    int d;

    for (d = 0; d < Dim; d++) {
        h ^= (unsigned int) Cell[d];
        h *= 16777619u;
    }

    return h;
}

int *IritGeomMergeFindSimilarPoints(IrtRType *VecPts,
				    int	PtLen,
				    int	VecLen,
				    IrtRType Tolerance)
{
    int i, j, *RetVecIdx, *BucketHead, *NextInBucket,
        HashSize = 16;
    IrtRType *R,
	TolSqr = IRIT_SQR(Tolerance);
    GMMergeSimilarPointsStruct *VecPtsData;

#ifdef IRIT_DEBUG_GM_PTS
    fprintf(stderr, "Before point merge:\n");
    for (i = 0, R = VecPts; i < VecLen; i++, R += PtLen) {
	if (IRIT_APX_EQ_EPS(R[0], 0.343, 1e-2) &&
	    IRIT_APX_EQ_EPS(R[1], -4.771, 1e-2) &&
	    IRIT_APX_EQ_EPS(R[2], 0.0321, 1e-2)) {
	    fprintf(stderr, "Pt %f  %f  %f\n", R[0], R[1], R[2]);
	}
    }
#endif /* IRIT_DEBUG_GM_PTS */

    VecPtsData = (GMMergeSimilarPointsStruct *)
                      IritMalloc(sizeof(GMMergeSimilarPointsStruct) * VecLen);
    RetVecIdx = (int *) IritMalloc(sizeof(int) * VecLen);

    for (i = 0, R = VecPts; i < VecLen; i++, R += PtLen) {
        VecPtsData[i].Idx = i;
        VecPtsData[i].Pt = R;

	RetVecIdx[i] = -1;
    }

    /* Points are bucketed into a uniform spatial hash grid, with cells    */
    /* sized to the merge tolerance.  This keeps the expected bucket       */
    /* occupancy O(1) even when many points share an axis value (e.g. on  */
    /* the shared, axis aligned faces of a tiled/lattice structure), which */
    /* is exactly the case that made the old X-sorted sweep degrade to    */
    /* O(n^2). */
    if (VecLen > 0 && Tolerance > 0.0) {
        IrtRType
	    InvTol = 1.0 / Tolerance;
        int NumNeighbors = 1;

        while (HashSize < VecLen * 2)
	    HashSize <<= 1;

        BucketHead = (int *) IritMalloc(sizeof(int) * HashSize);
        NextInBucket = (int *) IritMalloc(sizeof(int) * VecLen);
        for (i = 0; i < HashSize; i++)
	    BucketHead[i] = -1;

        for (i = 0; i < PtLen; i++)
	    NumNeighbors *= 3;

        for (i = 0; i < VecLen; i++) {
	    int Cell[PtLen], d;

	    for (d = 0; d < PtLen; d++)
	        Cell[d] = (int) floor(VecPtsData[i].Pt[d] * InvTol);

	    j = (int) (GMMergeHashCell(Cell, PtLen) & (unsigned int) (HashSize - 1));
	    NextInBucket[i] = BucketHead[j];
	    BucketHead[j] = i;
	}

        for (i = 0; i < VecLen; i++) {
	    int BaseCell[PtLen],
	        NeighborCell[PtLen],
	        MinIdx, NumIdenticalPts, n, d;

	    if (RetVecIdx[i] != -1)
	        continue;			      /* Already merged in. */

	    RetVecIdx[i] = i;
	    MinIdx = VecPtsData[i].Idx;
	    NumIdenticalPts = 1;

	    for (d = 0; d < PtLen; d++)
	        BaseCell[d] = (int) floor(VecPtsData[i].Pt[d] * InvTol);

	    for (n = 0; n < NumNeighbors; n++) {
	        int t = n, k;

	        for (d = 0; d < PtLen; d++) {
		    NeighborCell[d] = BaseCell[d] + (t % 3) - 1;
		    t /= 3;
		}

	        j = (int) (GMMergeHashCell(NeighborCell, PtLen) &
			   (unsigned int) (HashSize - 1));

	        for (k = BucketHead[j]; k != -1; k = NextInBucket[k]) {
		    if (k == i || RetVecIdx[k] != -1)
		        continue;

		    if (GMMergeComputePointsDistSqr(PtLen, &VecPtsData[i],
						     &VecPtsData[k]) < TolSqr) {
		        RetVecIdx[k] = i;
		        if (VecPtsData[k].Idx < MinIdx)
			    MinIdx = VecPtsData[k].Idx;
		        NumIdenticalPts++;
		    }
		}
	    }

	    if (NumIdenticalPts > 1) {
	        VecPtsData[i].Idx = MinIdx;

	        for (n = 0; n < NumNeighbors; n++) {
		    int t = n, k;

		    for (d = 0; d < PtLen; d++) {
		        NeighborCell[d] = BaseCell[d] + (t % 3) - 1;
			t /= 3;
		    }

		    j = (int) (GMMergeHashCell(NeighborCell, PtLen) &
			       (unsigned int) (HashSize - 1));

		    for (k = BucketHead[j]; k != -1; k = NextInBucket[k]) {
		        if (k != i && RetVecIdx[k] == i)
			    VecPtsData[k].Idx = MinIdx;
		    }
		}
	    }
	}

        IritFree(BucketHead);
        IritFree(NextInBucket);
    }

    /* Copy the indices. */
    for (i = 0; i < VecLen; i++) {
	int Idx = i;

	while (VecPtsData[Idx].Idx != Idx)
	    Idx = VecPtsData[Idx].Idx;
	RetVecIdx[i] = Idx;
    }

#ifdef IRIT_DEBUG_GM_PTS
    fprintf(stderr, "After  point merge:\n");
    for (i = 0; i < VecLen; i++) {
	GMMergeSimilarPointsStruct
	    *Ref = &VecPtsData[i];

	R = Ref -> Pt;
	fprintf(stderr, "Pt %d = %f  %f  %f (Idx = %d)\n",
		i, R[0], R[1], R[2], Ref -> Idx);
    }

    if (PtLen == 3) {
        /* Check we have no duplicated locations. */
        for (i = j = 0; i < VecLen; i++) {
	    if (RetVecIdx[i] == i)
	        j++;
	}
	fprintf(stderr, "%d unique points detected (out of %d), Tol %f,, Pt Size %d\n",
		j, VecLen, Tolerance, PtLen);
    }
#endif /* IRIT_DEBUG_GM_PTS */

#ifdef DEBUG
    if (PtLen == 3) {
	/* Exhaustive search for similar points. A slow n^2 test... */
	for (i = 0; i < VecLen; i++) {
	    IrtRType *R1, *R2;

	    R1 = &VecPts[i * 3];
	    for (j = i + 1; j < VecLen; j++) {
		R2 = &VecPts[j * 3];
		if (IRIT_PT_PT_DIST_SQR(R1, R2) < TolSqr &&
		    VecPtsData[i].Idx != VecPtsData[j].Idx) {
		    fprintf(stderr,
			    "Error: Same pt %d and %d: %f %f %f (Err=%f)\n",
			    i, j, R1[0], R1[1], R1[2],
			    IRIT_PT_PT_DIST(R1, R2));
		}
	    }
	}
    }
#endif /* DEBUG */

    IritFree(VecPtsData);

    return RetVecIdx;
}


