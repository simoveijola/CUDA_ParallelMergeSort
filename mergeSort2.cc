#include <bits/stdc++.h>
#include <immintrin.h>

// typedef unsigned long long data_t;
// typedef unsigned long long data8_t __attribute__ ((vector_size (8 * sizeof(unsigned long long))));
static constexpr int nd = 8;

static int nb = 5000000;
constexpr int n_threads = 8*16;

using namespace std;

// array, end and start node of first and second array respectively, length of shorter array, diagonal index array
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

template< class T >
static inline void find_path(T *arr, int s1, int s2, int pathlen, int m, int r, int num, pair<int,int> *ep) {
  int step = (pathlen + num - 1) / num;
  ep[0] = make_pair(s1, s2);

  #pragma omp parallel for num_threads(16) schedule(dynamic)
  for (int t = 1; t <= n_threads; ++t) {
    int diag = min(t * step, pathlen);
    
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

    ep[t] = make_pair(start1 + id1, start2 + id2);
    
  }
}

template< class T >
void merge_path(T *arr, int l, int m, int r, T *tmp, pair<int,int>* ep) {
  // sizes of arrays to merge
  if(arr[l] >= arr[r]) {
    copy(arr+m, arr+r, tmp);
    copy(arr+l, arr+m, tmp+r-m);
    return;
  } 
  if(arr[m+1] >= arr[m]) {
    copy(arr+l, arr+m, tmp);
    copy(arr+m, arr+r, tmp+m);
    return;
  }

  int n1 = m - l + 1;
  int n2 = r - m;

  int len = n1 + n2;
  int it = (len + nb - 1) / nb;

  int s1 = l;
  int s2 = m + 1;
  
  int sk = l;

  for (int i = 0; i < it; ++i) {
    int m1 = m - s1 + 1;
    int m2 = r - s2 + 1;
    int length = min(nb, m1 + m2);

    int lpt = (length + n_threads - 1) / n_threads;

    find_path(arr, s1, s2, lenght, m, r, n_threads, ep);

    #pragma omp parallel for schedule(dynamic)
    for(int t = 1; t <= n_threads; ++t) {
        if((t-1)*lpt > length) continue;

        merge(arr + ep[t - 1].first, arr + ep[t].first,
                    arr + ep[t - 1].second, arr + ep[t].second,
                    tmp + sk + (t-1) * lpt);
    }
    
    s1 = ep[n_threads].first;
    s2 = ep[n_threads].second;
    sk += length;
  }

}

template< class T >
void psort(int n, T *data) {
    // helper container to place sorted items into
    T *tmp = new T[n];
    // memory for the endpoints used during merge (n_threads intervals <=> n_threads + 1 end points)
    pair<int,int> *ep = new pair<int,int>[n_threads+1];

    int start_sz = (n+128-1)/128;
    
    if(start_sz > 1) {
        #pragma omp parallel for
        for (int start=0; start<n-1; start += start_sz) {
            int end = min(start + start_sz, n);
            sort(data + start, data + end);
        }
    }
    
    int nof_it = log2(n/start_sz);
    int cs = start_sz;

    for(int i = 0; i <= nof_it; ++i) {
        int size = pow(2, i)*start_sz;
        
        for (int start=0; start <= n-1; start += 2*size) {
            int mid = min(start + size - 1, n-1);
            int end = min(start + 2*size - 1, n-1);
            
            merge_path(data, start, mid, end, tmp, ep);
        }
        swap(tmp, data);
    }
    
    if(nof_it % 2  == 0 && n != 0) {

      int step = (n + 19) / 20;
      #pragma omp parallel for
      for (int i = 0; i < n; i += step) {
        int end = min(i + step, n);
        copy(data + i, data + end, tmp+i);
      }
      
      swap(tmp, data);
    }

    delete[] tmp; 
    delete[] ep;
}