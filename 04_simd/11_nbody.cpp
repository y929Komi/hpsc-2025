#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], jei[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    jei[i] = i;
  }

  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);

  for(int i=0; i<N; i++) {
	//float rx = x[i] - x[j];
        //float ry = y[i] - y[j];
	float fxr = 0;
	float fyr = 0;
	__m512 ai = _mm512_set1_ps(i);
	__m512 j = _mm512_load_ps(jei);
	__mmask16 mask = _mm512_cmpneq_ps_mask(ai, j);
        //float r = std::sqrt(rx * rx + ry * ry);
	__m512 xi = _mm512_set1_ps(x[i]);
	__m512 yi = _mm512_set1_ps(y[i]);
	__m512 rx = _mm512_sub_ps(xi, xvec);
	__m512 ry = _mm512_sub_ps(yi, yvec);
	__m512 rx2 = _mm512_mul_ps(rx, rx);
	__m512 ry2 = _mm512_mul_ps(ry, ry);
	__m512 r2 = _mm512_add_ps(rx2, ry2);
	__m512 rsqrt = _mm512_rsqrt14_ps(r2);
	__m512 rs2 = _mm512_mul_ps(rsqrt, rsqrt);
	__m512 rs3 = _mm512_mul_ps(rs2, rsqrt);
	rx = _mm512_mul_ps(rx, mvec);
	rx = _mm512_mul_ps(rx, rs3);
	ry = _mm512_mul_ps(ry, mvec);
	ry = _mm512_mul_ps(ry, rs3);
	fxr = _mm512_mask_reduce_add_ps(mask, rx);
	fyr = _mm512_mask_reduce_add_ps(mask, ry);
	fx[i] -= fxr;
	fy[i] -= fyr;
        //fx[i] -= rx * m[j] / (r * r * r);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
