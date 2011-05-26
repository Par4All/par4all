#define __PIPS_SAC_MULADD(a,b,c) ((a)+(b)*(c))
#define RWBITS 128

#define SIMD_LOAD_V4SF(v,a) v=vload4(0,a)
#define SIMD_STORE_V4SF(v,a) vstore4(v,0,a)
#define SIMD_MULADDPS(vres,a,b,c) vres=(a)+(b)*(c)
#define SIMD_ZERO_V4SF(vres) vres=0
typedef float4 v4sf;
//#include "opencl-simd.hpp"

__kernel void Convolve(const __global  float * pInput,
                        __constant float * pFilter,
                        __global  float * pOutput,
                        const int nInWidth,
                        const int nFilterWidth)
{
	int nWidth = get_global_size(0);

	int xOut = get_global_id(0);
	int yOut = get_global_id(1);

	int xInTopLeft = xOut;
	int yInTopLeft = yOut;

	float sum = 0;
	float RED0[4];
	int idxFtmp, yIn, idxIntmp;
	//PIPS generated variable
	int idxOut0, r0;

l99998:
	for(r0 = 0; r0 <= nFilterWidth-1; r0 += 1) {
		idxFtmp = r0*nFilterWidth;

		yIn = yInTopLeft+r0;
		idxIntmp = __PIPS_SAC_MULADD(xInTopLeft, yIn, nInWidth);

		//PIPS generated variable
		//PIPS generated variable
		int c0, c1;
		//PIPS generated variable
		v4sf vec00_0, vec10_0, vec20_0;
		SIMD_ZERO_V4SF(vec00_0);

l99999:
		for(c0 = 0; c0 <= 4*((nFilterWidth)/4)-1; c0 += 4) {
			//PIPS:SAC generated v4sf vector(s)
			SIMD_LOAD_V4SF(vec10_0, &pFilter[idxFtmp+c0]);
			SIMD_LOAD_V4SF(vec20_0, &pInput[idxIntmp+c0]);
			SIMD_MULADDPS(vec00_0, vec00_0, vec10_0, vec20_0);
		}
		SIMD_STORE_V4SF(vec00_0, &RED0[0]);
		sum = sum+RED0[0]+RED0[1]+RED0[2]+RED0[3];
		for(c1 = 4*((nFilterWidth)/4); c1 <= nFilterWidth-1; c1 += 1)
			sum = __PIPS_SAC_MULADD(sum, pFilter[idxFtmp+c1], pInput[idxIntmp+c1]);
	}

	idxOut0 = __PIPS_SAC_MULADD(xOut, yOut, nWidth);
	pOutput[idxOut0] = sum;
}
