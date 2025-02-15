#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include "error_check.hu"

/*
        #   #  #    #  #####  #####  ####   #####   ###   ####    ###   #     #  #####  #####  #####  ####   #####
        #   #   #  #   #   #  #      #   #  #   #  #   #  #   #  #   #  ##   ##  #        #    #      #   #  #
        #####    #     #####  #####  ####   #####  #####  ####   #####  # # # #  #####    #    #####  ####   #####
        #   #    #     #      #      #   #  #      #   #  #   #  #   #  #  #  #  #        #    #      #   #      #
        #   #    #     #      #####  #   #  #      #   #  #   #  #   #  #     #  #####    #    #####  #   #  #####
*/
// MERGING
constexpr int threadsPerMergeBlock = 64;
// maximum number of blocks 
constexpr int maxMergeBlocks = 512*2;
constexpr int elemsPerThread_merge = 1;

// BLOCK SORTING
constexpr int elemsPerThread_blockSort = 4; // 8
constexpr int blockSortSize = 1024; // 4096

/*
        #   #  #####  #  #      #####
        #   #    #    #  #      #
        #   #    #    #  #      #####
        #   #    #    #  #          #
        #####    #    #  #####  #####
*/

// performs single-threaded binary-search with one thread inside
// share memory of 32 (warp size) elements
__device__ int singleThreadBinarySearch(volatile bool tmp[32], int last) {
    int j = last/2, hi = last, lo = 0;
    while(hi > lo) {
        if(tmp[j]) {
            hi = j;
        } else {
            lo = j+1;
        }
        j = (hi+lo)/2;
    }
    return j;
}

// performs single-threaded binary-search with one thread inside
// share memory of 32 (warp size) elements
template< class T >
__device__ int singleThreadBinarySearch_T(volatile T tmp[2*threadsPerMergeBlock*elemsPerThread_merge]) {
    int iy = elemsPerThread_merge*threadIdx.x - 1;
    int hi = elemsPerThread_merge*threadIdx.x, lo = 0, j = hi/2;

    // printf("ib = %i, iy = %i, hi = %i\n", blockIdx.x, iy, hi);

    while(hi > lo) {
        if(tmp[iy - j] < tmp[threadsPerMergeBlock*elemsPerThread_merge + j]) {
            hi = j;
        } else {
            lo = j+1;
        }
        j = (hi+lo)/2;
    }
    return j;
}


// swaps two elements of the array arr
template< class T >
__device__ void swap_elements_at(volatile T *arr, int i, int j) {
    T holder = arr[i];
    arr[i] = arr[j];
    arr[j] = holder;
    return;
}


template< class T >
void swap_ptrs(T **A, T **B) {
    T *C = *B;
    *B = *A;
    *A = C;
    return;
}

// roundup of a/b
static inline int div_roundup(int a, int b) {
    return (a+b-1)/b;
}

/*
        #####   ###   #####  #   #      #####   ###   ####   #####  #  #####  #  #####  #   #  #  #   #   ###
        #   #  #   #    #    #   #      #   #  #   #  #   #    #    #    #    #  #   #  ##  #  #  ##  #  #   
        #####  #####    #    #####      #####  #####  ####     #    #    #    #  #   #  # # #  #  # # #  #  ##
        #      #   #    #    #   #      #      #   #  #   #    #    #    #    #  #   #  #  ##  #  #  ##  #   #
        #      #   #    #    #   #      #      #   #  #   #    #    #    #    #  #####  #   #  #  #   #   ###
*/


template< class T >
__device__ void find_intersection(T *arr1, T *arr2, int iy, int ix, int length, int *coords, volatile bool tmp[32]) {

    int ia = threadIdx.x;

    int intervals = (length + 32 - 1)/32;
    __shared__ int j;
    int hi = intervals-1, lo = 0;

    // first, using one thread, perform binary search to find the interval of warp size (32)
    // that contains the intersection with the path
    if(ia == 0) {
        j = (intervals-1)/2;
        while(hi > lo) {
            int i = j*32;
            int last = min(31, length-1-i);
            int idx = i + last;
            
            // traverse the arrays in opposite directions
            float lastelem = arr1[iy - 1 - idx] < arr2[ix + idx];
            // binary search the tmp array with one thread
            if(lastelem) {
                hi = j;
            } else {
                lo = j+1;
            }
            j = (hi+lo)/2; 
        }
    }

    __syncthreads();

    // load the 32 compared elements from the global memory to shared memory
    int i = j*32;
    int idx = i + ia;
    if(idx < length) {
        // traverse the arrays in opposite directions
        tmp[ia] = arr1[iy - 1 - idx] < arr2[ix + idx];
    }

    // using one thread again, find the intersection among the 32 elements
    if(ia == 0) {
        // the intersection is found from this section, now use single-threaded binary search to find the exact location
        int k = singleThreadBinarySearch(tmp, min(32, length-i)); 
        // store intersection
        coords[0] = iy - i - k;
        coords[1] = ix + i + k;
    }
}
  
template< class T >
__device__ void partition(T *arr1, T *arr2, int n, int m, int *intersections, volatile bool tmp[32], int S, int block, int nBlocks) {

    int ix = max(S-n, 0);
    int iy = min(S, n);

    int length = min(m-ix, iy);

    find_intersection(arr1, arr2, iy, ix, length, intersections + 2*(block+1), tmp);

    if(block + threadIdx.x == 0) {
        intersections[0] = 0, intersections[1] = 0;
        intersections[(nBlocks+1)*2] = n, intersections[(nBlocks+1)*2+1] = m;
    }

}

  template< class T >
__global__ void partition_all(T *arr, int N, int size, int *intersections_all, int blocksPerMerge, int nMerges) {

  int ia = threadIdx.x, ib = blockIdx.x, nb = gridDim.x;

  __shared__ bool tmp[32];

  T *arr1 = NULL, *arr2 = NULL;
  int block, bid;

  // nMerges in total and blocksPerMerge block operate always on a single merge
  while(ib < nMerges*(blocksPerMerge-1)) {
    // block: the local merge on which we operate on
    block = ib / (blocksPerMerge-1);
    // bid: the block id on the local merge 
    bid = ib % (blocksPerMerge-1);

    // the intersections storing starting pointer for this local merge
    int *intersections = intersections_all + (blocksPerMerge+1)*2*block;

    // starting pointers of the two arrays to be merged
    arr1 = arr + block*size;
    // the half-way is found at index start of arr1 + number of merges times the size of mergeable subarrays
    arr2 = arr + nMerges*size + block*size;

    // n,m: sizes of the arrays to merge, where m may differentiate from size, if it is the last sorted subarray
    int n = size;
    int m = block < nMerges - 1 ? size : min(N - (nMerges*2*size-size), size);

    // S: start index for this block on the local merged subarray  
    int S = ((unsigned long long int) (bid+1)*(n+m)) / (blocksPerMerge);
    // call partition algorithm to store the path intersection point in intersections[block] respectively for each merge
    partition(arr1, arr2, n, m, intersections, tmp, S, bid, blocksPerMerge-1);

    // add stride and continue
    ib += nb;
  }
}

/*
        #     #  #####  ####    ###   #  #   #   ###
        ##   ##  #      #   #  #      #  ##  #  #
        # # # #  #####  ####   #  ##  #  # # #  #  ##
        #  #  #  #      #   #  #   #  #  #  ##  #   #
        #     #  #####  #   #   ###   #  #   #   ###
*/


template< class T >
__global__ void merge_all_kernel(T *arr, T *aux, int size, int *intersections_all, int blocksPerMerge, int nMerges) {
    int ia = threadIdx.x, ib = blockIdx.x, bs = blockDim.x;

    __shared__ T tmp[2*threadsPerMergeBlock*elemsPerThread_merge];
   // __shared__ int indices[threadsPerMergeBlock];

    // start and end indices for arr1 and arr2
    __shared__ int sy, sx, ex, ey;
    // intersection coordinates pointer for each block
    int* intersections;

    int block, bid;
    int j, dir, iy, ix;
    int n_elements;

    T *arr1 = NULL, *arr2 = NULL;

    // go through all the mergeable array pairs
    while(ib < nMerges*blocksPerMerge) { 
        
        __syncthreads();
        // block: the local merge on which we operate on
        block = ib / blocksPerMerge;
        // bid: the block id on the local merge 
        bid = ib % blocksPerMerge;
        // starting pointers of the two arrays to be merged
        arr1 = arr + block*size;
        // the half-way is found at index start of arr1 + number of merges times the size of mergeable subarrays
        arr2 = arr + nMerges*size + block*size;

        // start pointer for intersections on this merge
        intersections = intersections_all + (blocksPerMerge+1)*2*block;

        if (ia == 0) {
            sy = intersections[2*bid], ey = intersections[2*(bid+1)];
            sx = intersections[2*bid + 1], ex = intersections[2*(bid+1) + 1];
        }
        
        __syncthreads();
        int aux_idx = block*size*2 + sy+sx + elemsPerThread_merge*ia;

        int bsize;

        while(sx < ex && sy < ey) {
            
            // load data to tmp  
            bsize = min(min(ex-sx, ey-sy), bs*elemsPerThread_merge);
            n_elements = min(bsize - ia * elemsPerThread_merge, elemsPerThread_merge);

            for(int i = ia; i < bsize; i += bs) {
                tmp[i] = arr1[sy + i], tmp[i + elemsPerThread_merge*bs] = arr2[sx + i];
            }
            
            __syncthreads();

            if(n_elements > 0) {
                // now perform the single-threaded binary-search for each 
                // thread in parallel using single-threaded binary-search
                
                j = singleThreadBinarySearch_T(tmp);

                iy = elemsPerThread_merge*ia - j, ix = elemsPerThread_merge*bs + j;
            }
            
            __syncthreads();

            for(int i = 0; i < n_elements; i++) {
                if (tmp[iy] < tmp[ix]) {
                    aux[aux_idx+i] = tmp[iy];
                    iy++;
                } else {
                    aux[aux_idx+i] = tmp[ix];
                    ix++;
                }
            }

            __syncthreads();
            // how many elements were added
            if(ia == (bsize - 1)/elemsPerThread_merge) {
                sx += ix-bs*elemsPerThread_merge, sy += iy;
            }

            __syncthreads();
            aux_idx += bsize;
        }
        __syncthreads();
        
        aux_idx = block*size*2 + sy+sx + ia;
        for(int idx = sx + ia; idx < ex; idx += bs) {
            aux[aux_idx] = arr2[idx];
            aux_idx += bs;
        }
        for(int idx = sy + ia; idx < ey; idx += bs) {
            aux[aux_idx] = arr1[idx];
            aux_idx += bs;
        }
        ib += gridDim.x;
    }
}

// merges sorted subarrays of size 'size' of the array 'data' of size 'n' to array 'aux'
// in total roundup(n/(2*size)) number of merges
template < class T >
void merge_all(T *data, T *aux, int n, int size, int *partition_coords, cudaStream_t mergeStream = NULL, cudaStream_t copyStream = NULL) {

    // initialize streams if not given.
    bool destroyCopyStream = false;
    bool destroyMergeStream = false;

    if(copyStream == NULL) {
        CUDA_CHK(cudaStreamCreate(&copyStream));
        destroyCopyStream = true;
    }
    if(mergeStream == NULL) { 
        CUDA_CHK(cudaStreamCreate(&mergeStream));
        destroyMergeStream = true;
    }
    // number of blocks used (can be explored)
    const int nBlocks = min(maxMergeBlocks, (n+1023)/1024);

    // number of merges to be done, and the number of blocks used for each merge
    int nMerges = ((n+size-1)/(size))/2;
    // printf("nMerges = %i\n", nMerges);

    int blocksPerMerge = max((nBlocks + nMerges - 1)/nMerges, 2);

    //int elemsPerBlock = size/blocksPerMerge;
    //elemsPerThread_blockSort = min(4, elemsPerBlock);

    // asynchronously copy the tail of the data not concerning the merges
    if((n-nMerges*size*2) > 0) {
        CUDA_CHK(cudaMemcpyAsync(aux + nMerges*size*2, data + nMerges*size*2, (n-nMerges*size*2)*sizeof(T), cudaMemcpyDeviceToDevice, copyStream));
    }

    // for partition we reduce one block per merge if plausible.
    partition_all<<<min(nMerges*(blocksPerMerge-1), nBlocks), 32, 0, mergeStream>>>(data, n, size, partition_coords, blocksPerMerge, nMerges);
    CUDA_CHK(cudaDeviceSynchronize());

    // merge the array to aux using partition_coords    nMerges*(blocksPerMerge), nBlocks)

    merge_all_kernel<<<min(nMerges*(blocksPerMerge-1), nBlocks), threadsPerMergeBlock, 0, mergeStream>>>(data, aux, size, partition_coords, blocksPerMerge, nMerges);

    // synchronize streams
    CUDA_CHK(cudaDeviceSynchronize());

    // clean up
    
    if(destroyCopyStream) CUDA_CHK(cudaStreamDestroy(copyStream));
    if(destroyMergeStream) CUDA_CHK(cudaStreamDestroy(mergeStream));

    return;
}



/*
        ####   #      #####   #####  #   #      #####  #####  ####  #####
        #   #  #      #   #   #      #  #       #      #   #  #   #   #
        ####   #      #   #   #      ###        #####  #   #  ####    #
        #   #  #      #   #   #      #  #           #  #   #  #   #   #
        ####   #####  #####   #####  #   #      #####  #####  #   #   #
*/

template< class T >
__global__ void bitonic_sort_kernel(T *data, int n, int nBlocks) {

    int ia = threadIdx.x, bid = blockIdx.x, bsize = blockDim.x, idx, offset, last;
    //bool lessThan;

    __shared__ T tmp[blockSortSize];

    for(int ib = bid; ib < nBlocks; ib += gridDim.x) {
        __syncthreads();
        int global_start = ib*2*elemsPerThread_blockSort*bsize, global_end = min(n, (ib+1)*2*elemsPerThread_blockSort*bsize);
        int length = global_end-global_start;

        // load data
        for(int i = ia; i < length; i+=bsize) {
            tmp[i] = data[global_start + i];
        }

        //printf("2*elemsPerThread_blockSort*bsize = %i\n", 2*elemsPerThread_blockSort*bsize);
        // perform bitonic sort:
        __syncthreads();

        // sorting steps: starting_offset = size of the offset we start at
        for(int i = 0; (1 << i) < 2*elemsPerThread_blockSort*bsize; i++) { 
            // add the new step
            last = (1 << i+1)-1;
            //printf("last = %i\n", last);
            for(int k = ia; k < 2*elemsPerThread_blockSort*bsize; k+=bsize) {
                idx = ((k >> i) << i+1) + (k & ((1 << i)-1));

                offset = last - 2*(k & ((1 << i)-1));
                //printf ("step1: thread %i: idx = %i, offset = %i\n", k, idx, offset);
                if(idx + offset < length && tmp[idx] > tmp[idx + offset]) {
                    swap_elements_at(tmp, idx, idx+offset);
                }
            }

            for(int j = i-1; j >= 0; --j) { 
                offset = 1 << j;
                __syncthreads();
                
                for(int k = ia; k < 2*elemsPerThread_blockSort*bsize; k+=bsize) {
                    //lessThan = (k >> i) & 1 == 1;
                    idx = ((k >> j) << j+1) + (k & (offset-1));
                    //printf ("step2: thread %i: idx = %i, offset = %i\n", k, idx, offset);
                    if(idx + offset < length && tmp[idx] > tmp[idx + offset]) {
                        swap_elements_at(tmp, idx, idx+offset);
                    }
                }
            }

            __syncthreads();
        }

        for(int i = ia; i < length; i+=bsize) {
            data[global_start + i] = tmp[i];
        }
    }
    return;
}

template< class T >
int blocksort(T *data, int n) {
    int size = min(n, blockSortSize);
    int nThreads = div_roundup(size, 2*elemsPerThread_blockSort);
    int nBlocks = div_roundup(n, size);

    bitonic_sort_kernel <<<1024, 128>>> (data, n, nBlocks);
    
    return size;
}


/*
        #####  #####  ####   #####
        #      #   #  #   #    #
        #####  #   #  ####     #
            #  #   #  #   #    #
        #####  #####  #   #    #
*/


template< class T > 
int cuda_sort(T **d_data, T **d_tmp, int* partition_coords, int n) {

    cudaStream_t mergeStream = NULL, copyStream = NULL;
    CUDA_CHK(cudaStreamCreate(&mergeStream));
    CUDA_CHK(cudaStreamCreate(&copyStream));

    // do block sorting
    blocksort(*d_data, n);
    CUDA_CHK(cudaDeviceSynchronize());

    int iterations = 0;
    int size;
    //printf("no errors on blocksort\n");
    // do merges until one sorted array

    for(size = blockSortSize; size < n; size*=2) {
        //printf("start merging size %i\n", size);
        merge_all(*d_data, *d_tmp, n, size, partition_coords, mergeStream, copyStream);
        
        CUDA_CHK(cudaGetLastError());

        swap_ptrs(d_data, d_tmp);

        iterations++;
    
    }

    CUDA_CHK(cudaStreamDestroy(copyStream));
    CUDA_CHK(cudaStreamDestroy(mergeStream));

    return iterations;

}


// sorting for host data
int sort(int n, unsigned long long *data) {

    if(n <= 1) return 0;

    unsigned long long *d_data = NULL, *d_tmp = NULL;
    CUDA_CHK(cudaMalloc((void**)&d_data, n*sizeof(unsigned long long)));
    CUDA_CHK(cudaMalloc((void**)&d_tmp, n*sizeof(unsigned long long)));

    // allocate array for intersection points
    int *partition_coords = NULL;
    CUDA_CHK(cudaMalloc((void**)&partition_coords, 2*(maxMergeBlocks+1)*sizeof(int)));

    CUDA_CHK(cudaMemcpy(d_data, data, n*sizeof(unsigned long long), cudaMemcpyHostToDevice));

    int nof_iters = cuda_sort(&d_data, &d_tmp, partition_coords, n);

    CUDA_CHK(cudaMemcpy(data, d_data, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    CUDA_CHK(cudaFree(d_data));
    CUDA_CHK(cudaFree(d_tmp));
    CUDA_CHK(cudaFree(partition_coords));

    CUDA_CHK(cudaGetLastError());

    return 0;

}