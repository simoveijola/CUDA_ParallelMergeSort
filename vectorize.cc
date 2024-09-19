#include "bits/stdc++.h"
#include "immintrin.h"
#include "typedefs.h"
#include "omp.h"

#define pd8_infty _mm512_set1_pd(std::numeric_limits<double>::infinity()) 

// n: number of doubles in data "data"
// nv: number of vectors in vectorized data "target"
// vectorizes the data while also sorting each vector internally using insertion sort
void vectorize(const unsigned long long n, const unsigned long long nv, const double *data, pd8 *target) {

    #pragma omp for schedule(dynamic, 32)
    for(int i = 0; i < nv-1; ++i) {
        for(int j = 0; j < 8; ++j) {
            double point = data[i*8+j];
            int k = j-1;
            while(k >= 0 && point < target[i][k]) {
                target[i][k+1] = target[i][k];
                --k;
            }
            target[i][k+1] = point;
        }
    }

    pd8 last = pd8_infty;
    for(int i = 0; i < n%8; ++i) {
        last[i] = data[(nv-1)*8+i];
    }

    target[nv-1] = last;

}

// n: number of doubles in data "target"
// nv: number of vectors in vectorized data "data"
// devectorizes data so, that first elements of vectors are filled to the target first, then second, and so on.
void devectorize(const unsigned long long n, const unsigned long long nv, const pd8 *data, double *target) {

    // overflow after last fully filled vector
    int of = n % 8;

    #pragma omp for
    for(int j = 0; j < 8; ++j) {
        // find where we start the filling from for this dimension
        int reduce = j >= of;
        int start = std::min(of, j)*nv + std::max(0, j-of)*(nv-1);

        for(int i = 0; i < nv - reduce; ++i) {
            target[start + i] = data[i][j];
        }
    }

}