#include <bits/stdc++.h>
#include <immintrin.h>

using namespace std;

// array, end and start node of first and second array respectively, length of shorter array, diagonal index array

template<class T>
static inline void swap(T* &a, T* &b) {
  T* t = b;
  b = a;
  a = t;
  return; 
}

template< class T >
static inline int bin_search(T *arr, int s1, int s2, int len) {

  int lo = 0, hi = len, i = (hi+lo) / 2;
  int e2 = s2 + len-1;
  while (hi > lo) {
    lo = arr[e2 - i] > arr[s1 + i] ? i + 1 : lo;
    hi = arr[e2 - i] > arr[s1 + i] ? hi : i;
    i = (hi + lo) / 2;
  }
  return lo;
}

// num = number of intervals
// pathlen = length to split into intervals
template< class T >
__global__ void find_path_kernel(T *arr, int s1, int s2, int pathlen, int m, int r, int num, pair<int,int> *ep) {

  int ib = blockIdx.x
  int i = ib;

  if(i = 0) ep[0] = make_pair(s1, s2);

  int step = (pathlen + num - 1) / num;

  int diag = min(i * step, pathlen);
  
  int start1 = s1;
  int start2 = s2;
  int search_len = diag;

  if (s1 + diag > m) {
    start2 += (diag - (m+1-s1));
    search_len -= (diag - (m+1-s1));
  }

  if (s2 + diag > r) {
    start1 += (diag - (r+1-s2));
    search_len -= (diag - (r+1-s2));
  }
  
  int id1 = bin_search(arr, start1, start2, search_len);
  int id2 = search_len - id1;

  ep[i+1] = make_pair(start1 + id1, start2 + id2);

}

template< class T >
__global__ void merge_kernel(T* data, T* tmp, int lpt, int n, pair<int, int> *ep) {
  int ia = threadIdx.x; int ib = blockIdx.x; int bs = blockDim.x;
  int s1, s2, e1, e2, s3;
  if(ep == NULL) {
    s1 = ib*2*lpt, s2 = (ib*2+1)*lpt, e1 = s2-1; s2 = min((ib+1)*2*lpt-1, n);
  } else {
    pair<int,int> s = ep[ib], e = ep[ib+1];
    s1 = s.first, s2 = s.second, e1 = e.first, e2 = e.second;
  }
  // starting point of the sorted elements in tmp
  int s3 = ib*2*lpt

  int i = s1 + ia, k = s3 + ia;
  
  int hi = e2-1, lo = s2, j = (hi+lo)/2;
  while(i < e1) {
    while(hi > lo) {
      if(data[i] > data[j]) {
        lo = j+1; 
      } else if(data[i] <= data[j]) {
        hi = j-1;
      }
    }
    tmp[k + lo] = data[i];
    i += bs, k += bs;
  }

  k = s3 + ia, i = s2 + ia;
  hi = e1-1, lo = s1, j = (hi+lo)/2;
  while(i < e2) {
    while(hi > lo) {
      if(data[i] >= data[j]) {
        lo = j+1; 
      } else if(data[i] < data[j]) {
        hi = j-1;
      }
    }
    tmp[k + lo] = data[i]; 
    i += bs, k += bs;
  }
}

template< class T >
void merge_path(T *arr, int l, int m, int r, T *tmp) {
   
  if(arr[l] >= arr[r]) {
    cudaMemcpy(tmp, arr+m+1, r-m, cudaMemcpyDeviceToDevice)
    cudaMemcpy(tmp+r-m, arr+l, m-l, cudaMemcpyDeviceToDevice)
    return;
  } 
  if(arr[m+1] >= arr[m]) {
    cudaMemcpy(tmp, arr+l, m-l, cudaMemcpyDeviceToDevice)
    cudaMemcpy(tmp+m, arr+m+1, m-l, cudaMemcpyDeviceToDevice)
    copy(arr+l, arr+m, tmp);
    copy(arr+m, arr+r, tmp+m);
    return;
  }

  int len = n1 + n2, s1 = l, s2 = m + 1;

  const int bs = 256;
  const int nb = (len+2*bs-1)/(2*bs);
  const int threadcount = nb*bs;
  const int pathlength = (length + nb - 1) / nb;

  pair<int,int> *ep = NULL;
  cudaMalloc((void**)&ep, (nb+1)*sizeof(pair<int,int>));

  find_path_kernel<<<nb, 1>>>(arr, s1, s2, len, m, r, nb, ep);
  merge_kernel<<<nb, bs>>>(arr, tmp, bs, n, ep);

}

void mergeSort(T *arr, T *tmp, int size, int n) {
  const bs = min(size, 256);
  const nb = (n+2*bs-1)/(2*bs);

  merge_kernel<<<nb, bs>>>(arr, tmp, size, n, NULL);
}

template< class T >
void psort_cuda(int n, T *data) {

    T* dataGPU = NULL;
    T* tmpGPU = NULL;
    pair<int,int> *ep = NULL;

    cudaMalloc((void**)&dataGPU, n * sizeof(T));
    cudaMalloc((void**)&tmpGPU, n * sizeof(T));
  
    cudaMemcpy(dataGPU, data, n*sizeof(T), cudaMemcpyHostToDevice);

    int size;
    // number of possible active block in GPU (needs to change according to GPU used)
    constexpr nb_capacity = 128;
    // tgt size tells us when we should switch to merge path (here the point when size of arrays to merge are such that we have nb_capacity arrays to merge)
    const int tgt_size = (n+nb_capacity-1)/(nb_capacity);
    
    for(size = 1; size < tgt_size; size*=2) {
      mergeSort(dataGPU, tmpGPU, size, n);
      swap(dataGPU, tmpGPU);
    }

    int nof_it = log2(n/size);
    int cs = size;

    for(int i = 0; i <= nof_it; ++i) {
        int sz = pow(2, i)*size;
        
        for (int start=0; start <= n-1; start += 2*size) {
            int mid = min(start + sz - 1, n-1);
            int end = min(start + 2*sz - 1, n-1);
            
            merge_path(dataGPU, start, mid, end, tmpGPU);
        }
        swap(tmpGPU, dataGPU);
    }

    cudaMemcpy(data, dataGPU, n*sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(dataGPU);
    cudaFree(tmpGPU);

}