#include <stdio.h>
#include <string.h>
#include "inc_irit/irit_sm.h"
#include "inc_irit/iritprsr.h"
#include "inc_irit/allocate.h"
#include "inc_irit/cagd_lib.h"
#include "inc_irit/triv_lib.h"

/* Build the same trivariate as UniformTilingCB and evaluate at every
   Greville abscissa (u_i, v_j, w_k).  Expected result: ctrl[i,j,k]. */

int main(void)
{
    int ULength = 3, VLength = 3, WLength = 3;
    int UOrder  = 2, VOrder  = 2, WOrder  = 2;
    int i, j, k, idx;

    /* Unique value per grid point: val = 100*i + 10*j + k
       so we can see exactly which (i,j,k) is being fetched */
    double src[27];
    {
        int ii, jj, kk, ff = 0;
        for (ii = 0; ii < ULength; ii++)
            for (jj = 0; jj < VLength; jj++)
                for (kk = 0; kk < WLength; kk++)
                    src[ff++] = 100*ii + 10*jj + kk;
    }
    int n = ULength * VLength * WLength;

    IritTrivTVStruct *TV = IritTrivBspTVNew(
        ULength, VLength, WLength,
        UOrder, VOrder, WOrder,
        IRIT_CAGD_PT_E1_TYPE
    );

    /* Build knot vectors (same logic as model_lin.c) */
    int knotU = ULength + UOrder;
    for (i = 0; i < knotU; i++) {
        if      (i < UOrder)   TV->UKnotVector[i] = 0.0;
        else if (i >= ULength) TV->UKnotVector[i] = 1.0;
        else TV->UKnotVector[i] = (double)(i - UOrder + 1) / (ULength - UOrder + 1);
    }
    int knotV = VLength + VOrder;
    for (i = 0; i < knotV; i++) {
        if      (i < VOrder)   TV->VKnotVector[i] = 0.0;
        else if (i >= VLength) TV->VKnotVector[i] = 1.0;
        else TV->VKnotVector[i] = (double)(i - VOrder + 1) / (VLength - VOrder + 1);
    }
    int knotW = WLength + WOrder;
    for (i = 0; i < knotW; i++) {
        if      (i < WOrder)   TV->WKnotVector[i] = 0.0;
        else if (i >= WLength) TV->WKnotVector[i] = 1.0;
        else TV->WKnotVector[i] = (double)(i - WOrder + 1) / (WLength - WOrder + 1);
    }

    {
        int iu, iv, iw;
        for (iu = 0; iu < ULength; iu++)
            for (iv = 0; iv < VLength; iv++)
                for (iw = 0; iw < WLength; iw++) {
                    int src_idx  = iu * VLength * WLength + iv * WLength + iw;
                    int irit_idx = iw * VLength * ULength + iv * ULength + iu;
                    TV->Points[1][irit_idx] = src[src_idx];
                }
    }

    /* Greville abscissae for order-2: xi_i = KnotVector[i+1] */
    double uG[3], vG[3], wG[3];
    for (i = 0; i < ULength; i++) uG[i] = TV->UKnotVector[i + 1];
    for (j = 0; j < VLength; j++) vG[j] = TV->VKnotVector[j + 1];
    for (k = 0; k < WLength; k++) wG[k] = TV->WKnotVector[k + 1];

    printf("Knot U: ");
    for (i = 0; i < knotU; i++) printf("%.3f ", TV->UKnotVector[i]);
    printf("\nGreville U: ");
    for (i = 0; i < ULength; i++) printf("%.3f ", uG[i]);
    printf("\n\n");

    printf("%-6s %-6s %-6s %-12s %-12s %-10s\n",
           "u_idx", "v_idx", "w_idx", "ctrl_val", "eval_val", "match");

    int flat_idx = 0;
    int errors = 0;
    for (i = 0; i < ULength; i++) {
        for (j = 0; j < VLength; j++) {
            for (k = 0; k < WLength; k++) {
                double expected = src[flat_idx++];
                CagdRType *res = IritTrivTVEval2Malloc(TV, uG[i], vG[j], wG[k]);
                double got = res[1];
                int match = (fabs(got - expected) < 1e-10);
                if (!match) errors++;
                printf("%-6d %-6d %-6d %-12.6f %-12.6f %s\n",
                       i, j, k, expected, got, match ? "OK" : "MISMATCH");
                IritFree(res);
            }
        }
    }

    printf("\n%s\n", errors == 0 ? "All values match." : "ERRORS FOUND.");
    IritTrivTVFree(TV);
    return errors;
}
