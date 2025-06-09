#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <iostream>

using namespace std;
typedef vector<vector<float>> matrix;
constexpr int BS = 8;

__global__ void navier_calc_b(float* u, float* v, float* p, float* b, float* pn, double rho, double dt, double dx, double dy, int nx, int ny)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= ny - 1 || col >= nx - 1)
    return;

  if (row > 0 && col > 0)
  {
    b[row * nx + col] = rho * (1 / dt *\
    ((u[row * nx + col + 1] - u[row * nx + col -1]) / (2 * dx) + (v[(row + 1) * nx + col] - v[(row - 1) * nx + col]) / (2 * dy)) -\
  (((u[row * nx + col + 1] - u[row * nx + col - 1]) / (2 * dx)) * ((u[row * nx + col + 1] - u[row * nx + col - 1]) / (2 * dx))) - 2 * ((u[(row + 1) * nx + col] - u[(row - 1) * nx + col]) / (2 * dy) * (v[row * nx + col + 1] - v[row * nx + col - 1]) / (2 * dx)) -\
(((v[(row + 1) * nx + col] - v[(row - 1) * nx + col]) / (2 * dy)) * ((v[(row + 1) * nx + col] - v[(row - 1) * nx + col]) / (2 * dy))));
  }
}

__global__ void navier_copy_p(float* p, float* pn, int nx, int ny)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= ny - 1 || col >= nx - 1)
    return;

  pn[row * nx + col] = p[row * nx + col];
}

__global__ void navier_calc_p(float* p, float* b, float* pn, double dx, double dy, int nx, int ny)
{

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= ny - 1 || col >= nx - 1)
    return;

  p[row * nx + col] = (dy * dy * (pn[row * nx + col + 1] + pn[row * nx + col - 1]) +\
dx * dx * (pn[(row + 1) * nx + col] + pn[(row - 1) * nx + col]) -\
b[row * nx + col] * dx * dx * dy * dy) / (2 * (dx * dx + dy * dy));
}

__global__ void navier_calc_pinit(float* p, int nx, int ny)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= ny - 1 || col >= nx - 1)
    return;

  if (col == 0) {
    p[row * nx] = p[row * nx + 1];
  }
  if (col == nx - 1) {
    p[row * nx + nx - 1] = p[row * nx + nx - 2];
  }
  if (row == 0) {
    p[col] = p[nx + col];
  }
  if (row == ny - 1) {
    p[(ny - 1) * nx + col] = 0;
  }
}

__global__ void navier_calc_uvcopy(float* u, float* v, float* un, float* vn, int nx, int ny)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= ny - 1 || col >= nx - 1)
    return;

  un[row * nx + col] = u[row * nx + col];
  vn[row * nx + col] = v[row * nx + col];
}

__global__ void navier_calc_uv(float* u, float* v, float* p, float* un, float* vn, double rho, double dt, double dx, double dy, double nu, int nx, int ny)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= ny - 1 || col >= nx - 1)
    return;

  u[row * nx + col] = un[row * nx + col] - un[row * nx + col] * (dt / dx) * (un[row * nx + col] - un[row * nx + col - 1]) - un[row * nx + col] * (dt / dy) * (un[row * nx + col] - un[(row - 1) * nx + col]) - dt / (2 * rho * dx) * (p[row * nx + col + 1] - p[row * nx + col - 1]) + nu * dt / (dx * dx) * (un[row * nx + col + 1] - 2 * un[row * nx + col] + un[row * nx + col - 1]) + nu * dt / (dy * dy) * (un[(row + 1) * nx + col] - 2 * un[row * nx + col] + un[(row - 1) * nx + col]);
  v[row * nx + col] = vn[row * nx + col] - vn[row * nx + col] * (dt / dx) * (vn[row * nx + col] - vn[row * nx + col - 1]) - vn[row * nx + col] * (dt / dy) * (vn[row * nx + col] - vn[(row - 1) * nx + col]) - dt / (2 * rho * dx) * (p[(row + 1) * nx + col] - p[(row - 1) * nx + col]) + nu * dt / (dx * dx) * (vn[row * nx + col + 1] - 2 * vn[row * nx + col] + vn[row * nx + col - 1]) + nu * dt / (dy * dy) * (vn[(row + 1) * nx + col] - 2 * vn[row * nx + col] + vn[(row - 1) * nx + col]);
}

__global__ void navier_calc_uvinit(float* u, float* v, int nx, int ny)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= ny - 1 || col >= nx - 1)
    return;

  if (col == 0) {
    u[row * nx] = 0;
    v[row * nx] = 0;
  }
  if (col == nx - 1) {
    u[row * nx + nx - 1] = 0;
    v[row * nx + nx - 1] = 0;
  }
  if (row == 0) {
    u[col] = 0;
    v[col] = 0;
  }
  if (row == ny - 1) {
    u[(ny - 1) * nx + col] = 1;
    v[(ny - 1) * nx + col] = 0;
  }
}
/*
__global__ void navier(float *u, float *v, float *p, float *b, float* un, float* vn, float*pn, int nx, int ny, double rho, double dt, double dx, double dy, int nit, double nu)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= ny - 1 || col >= nx - 1)
    return;

  navier_calc_b(u, v, p, b, pn, rho, dt, dx, dy, nx);

  for (int it = 0; it < nit; it++)
  {
    navier_copy_p(p, pn, nx);

    navier_calc_p(p, b, pn, dx, dy, nx);

    navier_calc_pinit(p, nx, ny);

    navier_calc_uvcopy(u, v, un, vn, nx);

    navier_calc_uv(u, v, pn, un, vn, rho, dt, dx, dy, nu, nx);

    navier_calc_uvinit(u, v, nx, ny);
  }
}
*/

void to_matrix_host(float *u_arr, float *v_arr, float *p_arr, float *b_arr, matrix u, matrix v, matrix p, matrix b, int nx, int ny)
{
  for (int j = 0; j < ny; ++j)
  {
    for (int i = 0; i < nx; ++i)
    {
      u[j][i] = u_arr[j * nx + i];
      v[j][i] = v_arr[j * nx + i];
      p[j][i] = p_arr[j * nx + i];
      b[j][i] = b_arr[j * nx + i];
    }
  }
}

void to_arr_host(matrix u, matrix v, matrix p, matrix b, float *u_arr, float *v_arr, float *p_arr, float *b_arr, int nx, int ny)
{
  for (int j = 0; j < ny; ++j)
  {
    for (int i = 0; i < nx; ++i)
    {
      u_arr[j * nx + i] = u[j][i];
      v_arr[j * nx + i] = v[j][i];
      p_arr[j * nx + i] = p[j][i];
      b_arr[j * nx + i] = b[j][i];
    }
  }
}

int main()
{
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  matrix u(ny, vector<float>(nx));
  matrix v(ny, vector<float>(nx));
  matrix p(ny, vector<float>(nx));
  matrix b(ny, vector<float>(nx));

  for (int j = 0; j < ny; j++)
  {
    for (int i = 0; i < nx; i++)
    {
      u[j][i] = 0;
      v[j][i] = 0;
      p[j][i] = 0;
      b[j][i] = 0;
    }
  }

  ofstream ufile("u_cu.dat");
  ofstream vfile("v_cu.dat");
  ofstream pfile("p_cu.dat");

  float *u_1d = (float *)calloc(nx * ny, sizeof(float));
  float *v_1d = (float *)calloc(nx * ny, sizeof(float));
  float *p_1d = (float *)calloc(nx * ny, sizeof(float));
  float *b_1d = (float *)calloc(nx * ny, sizeof(float));

  to_arr_host(u, v, p, b, u_1d, v_1d, p_1d, b_1d, nx, ny);

  float *u_gpu, *v_gpu, *p_gpu, *b_gpu, *un, *vn, *pn;

  cudaMalloc((void **)&u_gpu, sizeof(float) * nx * ny);
  cudaMalloc((void **)&v_gpu, sizeof(float) * nx * ny);
  cudaMalloc((void **)&p_gpu, sizeof(float) * nx * ny);
  cudaMalloc((void **)&b_gpu, sizeof(float) * nx * ny);
  cudaMalloc((void **)&un, sizeof(float) * nx * ny);
  cudaMalloc((void **)&vn, sizeof(float) * nx * ny);
  cudaMalloc((void **)&pn, sizeof(float) * nx * ny);

  cudaMemcpy(u_gpu, u_1d, sizeof(float) * nx * ny, cudaMemcpyHostToDevice);
  cudaMemcpy(v_gpu, v_1d, sizeof(float) * nx * ny, cudaMemcpyHostToDevice);
  cudaMemcpy(p_gpu, p_1d, sizeof(float) * nx * ny, cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, b_1d, sizeof(float) * nx * ny, cudaMemcpyHostToDevice);

  dim3 grid((nx + BS - 1) / BS, (ny + BS - 1) / BS, 1);
  dim3 block(BS, BS, 1);

  for (int n = 0; n < nt; ++n)
  {
    
    navier_calc_b<<<grid, block>>>(u_gpu, v_gpu, p_gpu, b_gpu, pn, rho, dt, dx, dy, nx, ny);
    cudaDeviceSynchronize();

  for (int it = 0; it < nit; it++)
  {
    navier_copy_p<<<grid, block>>>(p_gpu, pn, nx, ny);
    cudaDeviceSynchronize();

    navier_calc_p<<<grid, block>>>(p_gpu, b_gpu, pn, dx, dy, nx, ny);
    cudaDeviceSynchronize();

    navier_calc_pinit<<<grid, block>>>(p_gpu, nx, ny);
    cudaDeviceSynchronize();

    navier_calc_uvcopy<<<grid, block>>>(u_gpu, v_gpu, un, vn, nx, ny);
    cudaDeviceSynchronize();

    navier_calc_uv<<<grid, block>>>(u_gpu, v_gpu, pn, un, vn, rho, dt, dx, dy, nu, nx, ny);
    cudaDeviceSynchronize();

    navier_calc_uvinit<<<grid, block>>>(u_gpu, v_gpu, nx, ny);
    cudaDeviceSynchronize();
  }

    if (n % 10 == 0)
    {
   
      cudaMemcpy(u_1d, u_gpu, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);
      cudaMemcpy(v_1d, v_gpu, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);
      cudaMemcpy(p_1d, p_gpu, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);
      cudaMemcpy(b_1d, b_gpu, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);

      to_matrix_host(u_1d, v_1d, p_1d, b_1d, u, v, p, b, nx, ny);

      /*for (int i = 0; i < nx * ny; ++i) {
        cout << u_1d[i] << " ";
        if (i % nx == nx - 1) cout << "\n";
      }*/

      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          ufile << u[j][i] << " ";
      ufile << "\n";
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          vfile << v[j][i] << " ";
      vfile << "\n";
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          pfile << p[j][i] << " ";
      pfile << "\n";
    }
  }
  
  cudaMemcpy(u_1d, u_gpu, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);
  cudaMemcpy(v_1d, v_gpu, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);
  cudaMemcpy(p_1d, p_gpu, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);
  cudaMemcpy(b_1d, b_gpu, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);

  to_matrix_host(u_1d, v_1d, p_1d, b_1d, u, v, p, b, nx, ny);
  

  free(u_1d);
  free(v_1d);
  free(p_1d);
  free(b_1d);

  ufile.close();
  vfile.close();
  pfile.close();
}
