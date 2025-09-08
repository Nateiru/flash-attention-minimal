// C[M][N] = A[M][K] @ B[K][N]

// naive
__global__ void sgemm_v0(float* A, float* B, float* C, int M, int N, int K) {
  const int gx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gy = blockIdx.y * blockDim.y + threadIdx.y;

  if (gx < M && gy < N) {
    float tmp = 0;
    for (int i = 0; i < K; i++) {
      tmp += A[gx * K + i] * B[i * N + gy]; // C[gx][gy] = A[gx][i] * B[i][gy]
    }
    C[gx * N + gy] = tmp;
  }
}

// memory coalescing
__global__ void sgemm_v1(float* A, float* B, float* C, int M, int N, int K) {
  const int gy = blockIdx.x * blockDim.x + threadIdx.x; // 连续
  const int gx = blockIdx.y * blockDim.y + threadIdx.y;

  if (gx < M && gy < N) {
    float tmp = 0;
    for (int i = 0; i < K; i++) {
      tmp += A[gx * K + i] * B[i * N + gy]; // C[gx][gy] = A[gx][i] * B[i][gy]
    }
    C[gx * N + gy] = tmp;
  }
}

// block tile
template<const int BLOCK_SIZE>
__global__ void sgemm_v2(float* A, float* B, float* C, int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  const int BM = BLOCK_SIZE, BN = BLOCK_SIZE, BK = BLOCK_SIZE;

  int tx = threadIdx.x / BN; //  bank conflict
  int ty = threadIdx.x % BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 偏移到当前 block
  A = &A[bx * BM * K];
  B = &B[by * BN];
  C = &C[bx * BM * N + by * BN];

  float tmp = 0.0;

  for (int k = 0; k < K; k += BK) {
    As[tx * BK + tx] = A[tx * K + ty];
    Bs[tx * BN + tx] = B[tx * N + ty];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int i = 0; i < BK; i++) {
      tmp += As[tx * BK + i] * Bs[i * BN + ty];
    }
    __syncthreads();
  }

  C[tx * N + ty] = tmp;
}

// warp tile
// dim3 blockDim(256);
// dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
// sgemm_v3<128, 128, 8, 8, 8>
template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void sgemm_v3(float* A, float* B, float* C, int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int block_row_thread = BM / TM;
  int block_col_thread = BN / TN;
  int thread_num = block_row_thread * block_col_thread; // 一个 block 的线程数


  int tx = (threadIdx.x / block_col_thread) * TM;
  int ty = (threadIdx.x % block_col_thread) * TN;


  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 偏移到当前 block
  A = &A[bx * BM * K];
  B = &B[by * BN];
  C = &C[bx * BM * N + by * BN];

  int a_tile_row = threadIdx.x / BK;
  int a_tile_col = threadIdx.x % BK;
  int a_tile_stride = thread_num / BK;

  int b_tile_row = threadIdx.x / BN;
  int b_tile_col = threadIdx.x % BN;
  int b_tile_stride = thread_num / BN;


  float tmp[TM][TN] = {0.0f};

  for (int k = 0; k < K; k += BK) {
    // BM, BK
    for (int i = 0; i < BM; i += a_tile_stride) {
      As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
    }

    // BK, BN
    for (int i = 0; i < BK; i += b_tile_stride) {
      Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
    }

    __syncthreads();

    A += BK;
    B += BK * N;

    // 每个线程负责 TM TN 的计算
    for (int row = 0; row < TM; row++) {
      for (int col = 0; col < TN; col++) {
        for (int i = 0; i < BK; i++) {
          tmp[row][col] += As[(tx + row) * BK + i] * Bs[i * BN + (ty + col)];
        }
      }
    }
    __syncthreads();
  }

  for (int row = 0; row < TM; row++) {
    for (int col = 0; col < TN; col++) {
      C[(tx + row) * N + (ty + col)] = tmp[row][col];
    }
  }
}
