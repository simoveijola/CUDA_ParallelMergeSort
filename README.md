# ParallelMergeSort

Compatibility: 
g++10 10.5.0
nvcc 11.5, V11.5.119

## Testing
Build: nvcc -std=c++14 -ccbin=g++-10 -o test ./tests/test-merge.cu
Run ./test n  ,  n = array size

To compare with CUDA CUB toolkit mergesort, based on the same idea use 

nvcc -std=c++14 -ccbin=g++-10 -o test ./tests/test-merge.cu 

instead

## TODO:
- optimize hyperparameters
- modify merging to include also general rectangular windows for merging, only square windows currently
- optimize inside block sorting

