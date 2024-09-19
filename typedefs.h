#include <immintrin.h>

// Type definitions for different possible numerical vectors for both AVX and AVX-512

typedef unsigned long long ull8 __attribute__ ((vector_size (8 * sizeof(unsigned long long))));
typedef unsigned long long ull4 __attribute__ ((vector_size (4 * sizeof(unsigned long long))));

typedef long long ll8 __attribute__ ((vector_size (8 * sizeof(long long))));
typedef long long ll4 __attribute__ ((vector_size (4 * sizeof(long long))));

typedef unsigned int ui16 __attribute__ ((vector_size (16 * sizeof(unsigned int))));
typedef unsigned int ui8 __attribute__ ((vector_size (8 * sizeof(unsigned int))));

typedef int i16 __attribute__ ((vector_size (16 * sizeof(int))));
typedef int i8 __attribute__ ((vector_size (8 * sizeof(int))));

typedef unsigned short us32 __attribute__ ((vector_size (32*sizeof(unsigned short))));
typedef unsigned short us16 __attribute__ ((vector_size (16*sizeof(unsigned short))));

typedef short s32 __attribute__ ((vector_size (32*sizeof(short))));
typedef short s16 __attribute__ ((vector_size (16*sizeof(short))));

typedef unsigned char uc64 __attribute__ ((vector_size (64*sizeof(unsigned char))));
typedef unsigned char uc32 __attribute__ ((vector_size (32*sizeof(unsigned char))));

typedef char c64 __attribute__ ((vector_size (64*sizeof(char))));
typedef char c32 __attribute__ ((vector_size (32*sizeof(char))));

typedef double pd8 __attribute__ ((vector_size (8 * sizeof(double))));
typedef double pd4 __attribute__ ((vector_size (4 * sizeof(double))));

typedef float ps16 __attribute__ ((vector_size (16 * sizeof(float))));
typedef float ps8 __attribute__ ((vector_size (8 * sizeof(float))));

// decide the definitions used

#if defined(__AVX512F__)
    #define load_pd _mm512_load_pd
    #define load_ps _mm512_load_ps
    #define load_epi32 _mm512_load_epi32
    #define load_epi64 _mm512_load_epi64
    #define loadu_epi32 _mm512_loadu_epi32
    #define loadu_epi64 _mm512_loadu_epi64
    #define loadu_epi16 _mm512_loadu_epi16
    #define loadu_epi8 _mm512_loadu_epi8
    #define max_pd _mm512_max_pd
    #define min_pd _mm512_min_pd
#else
    #define load_pd _mm256_load_pd
    #define load_ps _mm256_load_ps
    #define load_epi32 _mm256_load_epi32
    #define load_epi64 _mm256_load_epi64
    #define loadu_epi32 _mm256_loadu_epi32
    #define loadu_epi64 _mm256_loadu_epi64
    #define loadu_epi16 _mm256_loadu_epi16
    #define loadu_epi8 _mm256_loadu_epi8
    #define max_pd _mm256_max_pd
#endif
