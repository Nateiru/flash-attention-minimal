// 1. 向上取整
#define CEIL(a, b) ((a + b - 1) / (b))

// 2. FLOAT4，用于向量化访存，c++写法
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// int block_size = 4;
// int grid_size = CEIL(N, block_size);
// elementwise_add<<<grid_size, block_size>>>(a, b, c, N);

__global__ void elementwise_add(float* a, float* b, float* c, int N) {
  int idx = blockIdx.x * blockDim.x  + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void elementwise_add_float4(float* a, float* b, float *c, int N) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 tmp_a = FLOAT4(a[idx]);
    float4 tmp_b = FLOAT4(b[idx]);
    float4 tmp_c;
    tmp_c.x = tmp_a.x + tmp_b.x;
    tmp_c.y = tmp_a.y + tmp_b.y;
    tmp_c.z = tmp_a.z + tmp_b.z;
    tmp_c.w = tmp_a.w + tmp_b.w;
    FLOAT4(c[idx]) = tmp_c;
  }
}

__global__ void sigmod(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = 1.0f / (1.0f + expf(-x[idx]));
  }
}

__global__ void relu(float* x, float* y, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) y[idx] = fmaxf(0.0f, x[idx]);
}
