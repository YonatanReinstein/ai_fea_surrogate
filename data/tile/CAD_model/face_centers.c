#include "face_centers.h"

int g_ni, g_nj, g_nk;
CagdRType *g_data;

int main(int argc, char *argv[]) {
    if (argc != 7) {
        fprintf(stderr, "Usage: %s i j k u v w\n", argv[0]);
        return 1;
    }

    g_ni = atoi(argv[1]); g_nj = atoi(argv[2]); g_nk = atoi(argv[3]);
    if (g_ni <= 0 || g_nj <= 0 || g_nk <= 0) {
        fprintf(stderr, "i, j, k must be positive integers\n");
        return 1;
    }

    int total = face_count(g_ni, g_nj, g_nk);
    g_data = malloc(total * sizeof(CagdRType));
    for (int i = 0; i < total; i++) g_data[i] = i * 0.1;

    CagdRType u = atof(argv[4]), v = atof(argv[5]), w = atof(argv[6]);
    printf("%.10g\n", UniformTilingCB(u, v, w));

    free(g_data);
    return 0;
}
