#include <cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>
#include "../single_file_mergesort.cu"
#include "../error_check.hu"

typedef unsigned long long ull;

float getElapsedTime(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}

std::vector<int> isSorted(ull *data, int n) {
    std::vector<int> ind;
    ull prev = data[0];
    for(int i = 1; i < n; ++i) {
        if(prev > data[i]) {
            ind.push_back(i);
        }
        prev = data[i];
    }
    return ind;
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    int roundedn = (n+1)/2;

    ull *data = (ull*) malloc(n*sizeof(ull));
    ull *aux = (ull*) malloc(n*sizeof(ull));
    ull *data2 = (ull*) malloc(n*sizeof(ull));

    std::srand(1223);
    for (int i = 0; i < n; ++i) {
        data[i] = static_cast<ull>(std::rand()); // Random int between 0 and 1
    }

    std::copy(data, data + n, data2);

    ull *d_data = NULL, *d_tmp = NULL, *tmp = NULL;
    
    CUDA_CHK(cudaMallocHost((void**)&tmp, n*sizeof(ull)));

    int size = min(n, 5000);

    CUDA_CHK(cudaMalloc((void**)&d_data, n*sizeof(unsigned long long)));
    CUDA_CHK(cudaMalloc((void**)&d_tmp, n*sizeof(unsigned long long)));

    CUDA_CHK(cudaMemcpy(d_data, data, n*sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // allocate array for intersection points
    int *partition_coords = NULL;
    int maxMergeBlocks = 1024;
    CUDA_CHK(cudaMalloc((void**)&partition_coords, 2*(maxMergeBlocks+1)*sizeof(int)));

    auto start = std::chrono::high_resolution_clock::now();
    std::sort(data, data + n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("Time taken in std::sort for size %i: %li microseconds \n", n, duration.count());

    start = std::chrono::high_resolution_clock::now();

    int nof_iters = cuda_sort(&d_data, &d_tmp, partition_coords, n);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("Time taken in cuda sort for size %i: %li microseconds \n", n, duration.count());

    cudaMemcpy(data2, d_data, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    CUDA_CHK(cudaGetLastError());
    
    CUDA_CHK(cudaDeviceSynchronize());

    std::cout << std::is_sorted(data2, data2 + n) << std::endl;

    printf("\n\n");

    CUDA_CHK(cudaFree(d_data));
    CUDA_CHK(cudaFree(d_tmp));
    CUDA_CHK(cudaFree(partition_coords));
    return 0;
}