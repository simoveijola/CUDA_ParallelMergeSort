
#include "./mergeSort_vectorized.cc"
// #include "mergeVectorized.cc"
#include "./mergeSort2.cc"
#include "bits/stdc++.h"
#include "execution"


using namespace std;
typedef double data_t;
#define MAX numeric_limits<data_t>::max()

int main() {

  unsigned long long n;
  cin >> n;
  data_t *data = new data_t[n];
  data_t *data2 = new data_t[n];

  default_random_engine eng;
  uniform_real_distribution<double> unif(0.0, 1.0);

  for(unsigned long long i = 0; i < n; ++i) {
    data[i] = (data_t) MAX*unif(eng);
    data2[i] = (data_t) MAX*unif(eng);
  }

  vector<data_t> cp = vector<data_t>(n);
  copy(data, data+n, cp.begin());
  
  cout << "begin" << endl;
  auto start = std::chrono::high_resolution_clock::now();
  // idea:
  psort(n, data);
  auto end = std::chrono::high_resolution_clock::now();
  cout << (float) std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "\n";

  bool isSorted = true;
  for(int i = 1; i < n; ++i) {
    isSorted = isSorted && (data[i] >= data[i-1]);
  }
  cout << isSorted << endl;

  start = std::chrono::high_resolution_clock::now();
  //x86simdsortStatic::qsort(data2, n);
  std::sort(std::execution::par_unseq, cp.begin(), cp.end()); // std::execution::par_unseq, 
  end = std::chrono::high_resolution_clock::now();
  cout << (float) std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "\n";

  // psortv(n, data2);

  /*
  pd8 a = {1., 3., 5., 7., 9., 11., 13., 15.};
  pd8 b = {15., 13., 11., 9., 7., 5., 3., 1.};
  pd8* A = new pd8[4]; A[0] = a; A[1] = b; A[2] = b; A[3] = a;
  pd8* C = new pd8[4];


  mergev(A, A+1, A+1, A+2, C);
  mergev(A+2, A+3, A+3, A+4, C+2);

  for(int i = 0; i < 8; ++i) {
    for(int j = 0; j < 2; ++j) {
      cout << C[j][i] << " ";
    }
  }

  cout << "\n";

  for(int i = 0; i < 8; ++i) {
    for(int j = 2; j < 4; ++j) {
      cout << C[j][i] << " ";
    }
  }

  cout << "\n";

  std::swap(A, C);
  mergev(A, A+2, A+2, A+4, C);

  std::swap(A, C);

  for(int i = 0; i < 8; ++i) {
    for(int j = 0; j < 4; ++j) {
      cout << A[j][i] << " ";
    }
  }

  cout << "\n";
  */
  
  return 0;
  
}