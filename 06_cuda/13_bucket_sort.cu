#include <cstdio>
#include <cstdlib>

__global__ void bucket_sort(int *vec, int *key, int *idx, int N, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j;
  if (i >= range) return;
  vec[i] = 0;
  idx[i] = 0;
  __syncthreads();
  if (i == 0) {
    for (int k = 0; k < N; ++k) {
        vec[key[k]]++;
    }
    for (int k = 0; k < range-1; ++k) {
        idx[k+1] = idx[k] + vec[k];
    }
  }
  __syncthreads();
    j = idx[i];

    for(; vec[i] > 0; --vec[i]) {
      key[j++] = i;
    }
    
}

int main() {
  int n = 50;
  int range = 5;
  const int BlockSize = 16;
  int *bucket, *key, *idx;
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&idx, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int numBlocks = (range + BlockSize - 1) / BlockSize;
  bucket_sort<<<numBlocks, BlockSize>>>(bucket, key, idx, n, range);

  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}

