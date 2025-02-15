#include <cub/device/device_merge_sort.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

typedef unsigned long long ull;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct CustomOpT {
    __device__ bool operator()(const ull& a, const ull& b) const {
        return a < b;  // Sorting in descending order
    }
};

int main() {
    const size_t N = 100000000;  // 100 million keys

    // Allocate device memory for keys
    ull* d_keys;
    CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(ull)));

    // Allocate host memory and fill with random values
    ull* h_keys = new ull[N];
    for (size_t i = 0; i < N; ++i) {
        h_keys[i] = static_cast<ull>(rand()) / RAND_MAX;
    }

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, N * sizeof(ull), cudaMemcpyHostToDevice));

    // CUB temporary storage
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CustomOpT custom_op;

    // Start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Step 1: Determine the size of temporary storage required
    cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, N, custom_op);

    // Step 2: Allocate the temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Step 3: Perform the merge sort on the device
    cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, N, custom_op);

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Sorting 100M floats took " << milliseconds << " ms with cuda cub mergesort\n";

    // Cleanup
    delete[] h_keys;
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_temp_storage));

    return 0;
}
