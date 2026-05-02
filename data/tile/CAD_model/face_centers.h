#ifndef FACE_CENTERS_H
#define FACE_CENTERS_H

#include "inc_irit/irit_sm.h"
#include "inc_irit/iritprsr.h"
#include "inc_irit/allocate.h"
#include "inc_irit/attribut.h"
#include "inc_irit/geom_lib.h"
#include "inc_irit/cagd_lib.h"
#include "inc_irit/user_lib.h"
#include "inc_irit/grap_lib.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern int g_ni, g_nj, g_nk;
extern CagdRType *g_data;

/*
 * For a unit cube divided into g_ni x g_nj x g_nk sub-cubes, face centers are:
 *   X-normal: x = a/ni,       y = (b+0.5)/nj,  z = (c+0.5)/nk
 *   Y-normal: x = (a+0.5)/ni, y = b/nj,         z = (c+0.5)/nk
 *   Z-normal: x = (a+0.5)/ni, y = (b+0.5)/nj,  z = c/nk
 *
 * Total unique face centers: (ni+1)*nj*nk + ni*(nj+1)*nk + ni*nj*(nk+1)
 */

static int face_count(int ni, int nj, int nk) {
    return (ni + 1) * nj * nk
         + ni * (nj + 1) * nk
         + ni * nj * (nk + 1);
}

/*
 * O(1) lookup: maps (u,v,w) face-center coordinates to the value stored in
 * g_data at that face's index. g_ni/g_nj/g_nk/g_data must be initialised
 * before the first call.
 */
static CagdRType UniformTilingCB(CagdRType u, CagdRType v, CagdRType w) {
    int ni = g_ni, nj = g_nj, nk = g_nk;
    const double eps = 1e-9;
#define NEAR_INT(val) (fabs((val) - round(val)) < eps)

    /* X-normal: u on integer grid, v and w on half grid */
    {
        double au = u * ni, bv = v * nj - 0.5, cw = w * nk - 0.5;
        int a = (int)round(au), b = (int)round(bv), c = (int)round(cw);
        if (NEAR_INT(au) && NEAR_INT(bv) && NEAR_INT(cw)
                && a >= 0 && a <= ni && b >= 0 && b < nj && c >= 0 && c < nk)
            return g_data[a*nj*nk + b*nk + c];
    }

    /* Y-normal: v on integer grid, u and w on half grid */
    {
        double au = u * ni - 0.5, bv = v * nj, cw = w * nk - 0.5;
        int a = (int)round(au), b = (int)round(bv), c = (int)round(cw);
        if (NEAR_INT(au) && NEAR_INT(bv) && NEAR_INT(cw)
                && a >= 0 && a < ni && b >= 0 && b <= nj && c >= 0 && c < nk)
            return g_data[(ni+1)*nj*nk + a*(nj+1)*nk + b*nk + c];
    }

    /* Z-normal: w on integer grid, u and v on half grid */
    {
        double au = u * ni - 0.5, bv = v * nj - 0.5, cw = w * nk;
        int a = (int)round(au), b = (int)round(bv), c = (int)round(cw);
        if (NEAR_INT(au) && NEAR_INT(bv) && NEAR_INT(cw)
                && a >= 0 && a < ni && b >= 0 && b < nj && c >= 0 && c <= nk)
            return g_data[(ni+1)*nj*nk + ni*(nj+1)*nk + a*nj*(nk+1) + b*(nk+1) + c];
    }

#undef NEAR_INT

    return -1.0;
}

#endif /* FACE_CENTERS_H */
