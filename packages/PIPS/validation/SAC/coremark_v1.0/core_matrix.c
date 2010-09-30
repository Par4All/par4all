/*
Author : Shay Gal-On, EEMBC

This file is part of  EEMBC(R) and CoreMark(TM), which are Copyright (C) 2009 
All rights reserved.                            

EEMBC CoreMark Software is a product of EEMBC and is provided under the terms of the
CoreMark License that is distributed with the official EEMBC COREMARK Software release. 
If you received this EEMBC CoreMark Software without the accompanying CoreMark License, 
you must discontinue use and download the official release from www.coremark.org.  

Also, if you are publicly displaying scores generated from the EEMBC CoreMark software, 
make sure that you are in compliance with Run and Reporting rules specified in the accompanying readme.txt file.

EEMBC 
4354 Town Center Blvd. Suite 114-200
El Dorado Hills, CA, 95762 
*/ 
#include "coremark.h"
/*
Topic: Description
	Matrix manipulation benchmark
	
	This very simple algorithm forms the basis of many more complex algorithms. 
	
	The tight inner loop is the focus of many optimizations (compiler as well as hardware based) 
	and is thus relevant for embedded processing. 
	
	The total available data space will be divided to 3 parts:
	NxN Matrix A - initialized with small values (upper 3/4 of the bits all zero).
	NxN Matrix B - initialized with medium values (upper half of the bits all zero).
	NxN Matrix C - used for the result.

	The actual values for A and B must be derived based on input that is not available at compile time.
*/
ee_s16 matrix_test(ee_u32 N, MATRES C[N][N], MATDAT A[N][N], MATDAT B[N][N], MATDAT val);
ee_s16 matrix_sum(ee_u32 N, MATRES C[N][N], MATDAT lipval);
void matrix_mul_const(ee_u32 N, MATRES C[N][N], MATDAT A[N][N], MATDAT al);
void matrix_mul_vect(ee_u32 N, MATRES C[N], MATDAT A[N][N], MATDAT B[N]);
void matrix_mul_matrix(ee_u32 N, MATRES C[N][N], MATDAT A[N][N], MATDAT B[N][N]);
void matrix_mul_matrix_bitextract(ee_u32 N, MATRES C[N][N], MATDAT A[N][N], MATDAT B[N][N]);
void matrix_add_const(ee_u32 N, MATDAT A[N][N], MATDAT val);

#define matrix_test_next(x) (x+1)
#define matrix_clip(x,y) ((y) ? (x) & 0x0ff : (x) & 0x0ffff)
#define matrix_big(x) (0xf000 | (x))
#define bit_extract(x,from,to) (((x)>>(from)) & (~(0xffffffff << (to))))

#if CORE_DEBUG
void printmat(ee_u32 N, MATDAT A[N][N], char *name) {
	ee_u32 i,j;
	ee_printf("Matrix %s [%dx%d]:\n",name,N,N);
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			if (j!=0)
				ee_printf(",");
			ee_printf("%d",A[i][j]);
		}
		ee_printf("\n");
	}
}
void printmatC(ee_u32 N, MATRES C[N][N], char *name) {
	ee_u32 i,j;
	ee_printf("Matrix %s [%dx%d]:\n",name,N,N);
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			if (j!=0)
				ee_printf(",");
			ee_printf("%d",C[i][j]);
		}
		ee_printf("\n");
	}
}
#endif
/* Function: core_bench_matrix
	Benchmark function

	Iterate <matrix_test> N times, 
	changing the matrix values slightly by a constant amount each time.
*/
ee_u16 core_bench_matrix(mat_params *p, ee_s16 seed, ee_u16 crc) {
	ee_u32 N=p->N;
	MATRES (*C)[N][N]=(MATRES (*) [N][N])p->C;
	MATDAT (*A)[N][N]=(MATDAT (*) [N][N])p->A;
	MATDAT (*B)[N][N]=(MATDAT (*) [N][N])p->B;
	MATDAT val=seed;

	crc=crc16(matrix_test(N,*C,*A,*B,val),crc);

	return crc;
}

/* Function: matrix_test
	Perform matrix manipulation.

	Parameters:
	N - Dimensions of the matrix.
	C - memory for result matrix.
	A - input matrix
	B - operator matrix (not changed during operations)

	Returns:
	A CRC value that captures all results calculated in the function.
	In particular, crc of the value calculated on the result matrix 
	after each step by <matrix_sum>.

	Operation:
	
	1 - Add a constant value to all elements of a matrix.
	2 - Multiply a matrix by a constant.
	3 - Multiply a matrix by a vector.
	4 - Multiply a matrix by a matrix.
	5 - Add a constant value to all elements of a matrix.

	After the last step, matrix A is back to original contents.
*/
ee_s16 matrix_test(ee_u32 N, MATRES C[N][N], MATDAT A[N][N], MATDAT B[N][N], MATDAT val) {
	ee_u16 crc=0;
	MATDAT clipval=matrix_big(val);

	matrix_add_const(N,A,val); /* make sure data changes  */
#if CORE_DEBUG
	printmat(N,A,"matrix_add_const");
#endif
	matrix_mul_const(N,C,A,val);
	crc=crc16(matrix_sum(N,C,clipval),crc);
#if CORE_DEBUG
	printmatC(N,C,"matrix_mul_const");
#endif
	matrix_mul_vect(N,C[0],A,B[0]);
	crc=crc16(matrix_sum(N,C,clipval),crc);
#if CORE_DEBUG
	printmatC(N,C,"matrix_mul_vect");
#endif
	matrix_mul_matrix(N,C,A,B);
	crc=crc16(matrix_sum(N,C,clipval),crc);
#if CORE_DEBUG
	printmatC(N,C,"matrix_mul_matrix");
#endif
	matrix_mul_matrix_bitextract(N,C,A,B);
	crc=crc16(matrix_sum(N,C,clipval),crc);
#if CORE_DEBUG
	printmatC(N,C,"matrix_mul_matrix_bitextract");
#endif
	
	matrix_add_const(N,A,-val); /* return matrix to initial value */
	return crc;
}

/* Function : matrix_init
	Initialize the memory block for matrix benchmarking.

	Parameters:
	blksize - Size of memory to be initialized.
	memblk - Pointer to memory block.
	seed - Actual values chosen depend on the seed parameter.
	p - pointers to <mat_params> containing initialized matrixes.

	Returns:
	Matrix dimensions.
	
	Note:
	The seed parameter MUST be supplied from a source that cannot be determined at compile time
*/
ee_u32 core_init_matrix(ee_u32 blksize, void *memblk, ee_s32 seed, mat_params *p) {
	ee_u32 N=0;
	ee_s32 order=1;
	MATDAT val;
	ee_u32 i=0,j=0;
	if (seed==0)
		seed=1;
	while (j<blksize) {
		i++;
		j=i*i*2*4;		
	}
	N=i-1;
    {
        MATDAT (*A)[N][N];
        MATDAT (*B)[N][N];
        A=(MATDAT (*)[N][N])align_mem(memblk);
        B=(MATDAT (*)[N][N])((MATDAT*)*A+N*N);

        for (i=0; i<N; i++) {
            for (j=0; j<N; j++) {
                seed = ( ( order * seed ) % 65536 );
                val = (seed + order);
                val=matrix_clip(val,0);
                (*B)[i][j] = val;
                val =  (val + order);
                val=matrix_clip(val,1);
                (*A)[i][j] = val;
                order++;
            }
        }

        p->A=(MATDAT*)&((*A)[0][0]);
        p->B=(MATDAT*)&((*B)[0][0]);
        p->C=(MATRES*)align_mem((MATDAT*)*B+N*N);
        p->N=N;
#if CORE_DEBUG
        printmat(N,A,"A");
        printmat(N,B,"B");
#endif
    }
	return N;
}

/* Function: matrix_sum
	Calculate a function that depends on the values of elements in the matrix.

	For each element, accumulate into a temporary variable.
	
	As long as this value is under the parameter clipval, 
	add 1 to the result if the element is bigger then the previous.
	
	Otherwise, reset the accumulator and add 10 to the result.
*/
ee_s16 matrix_sum(ee_u32 N, MATRES C[N][N], MATDAT clipval) {
	MATRES tmp=0,prev=0,cur=0;
	ee_s16 ret=0;
	ee_u32 i,j;
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			cur=C[i][j];
			tmp+=cur;
			if (tmp>clipval) {
				ret+=10;
				tmp=0;
			} else {
				ret += (cur>prev) ? 1 : 0;
			}
			prev=cur;
		}
	}
	return ret;
}

/* Function: matrix_mul_const
	Multiply a matrix by a constant.
	This could be used as a scaler for instance.
*/
void matrix_mul_const(ee_u32 N, MATRES C[N][N], MATDAT A[N][N], MATDAT val) {
	ee_u32 i,j;
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			C[i][j]=A[i][j] * val;
		}
	}
}

/* Function: matrix_add_const
	Add a constant value to all elements of a matrix.
*/
void matrix_add_const(ee_u32 N, MATDAT A[N][N], MATDAT val) {
	ee_u32 i,j;
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			A[i][j] += val;
		}
	}
}

/* Function: matrix_mul_vect
	Multiply a matrix by a vector.
	This is common in many simple filters (e.g. fir where a vector of coefficients is applied to the matrix.)
*/
void matrix_mul_vect(ee_u32 N, MATRES C[N], MATDAT A[N][N], MATDAT B[N]) {
	ee_u32 i,j;
	for (i=0; i<N; i++) {
		C[i]=0;
		for (j=0; j<N; j++) {
			C[i]+=A[i][j] * B[j];
		}
	}
}

/* Function: matrix_mul_matrix
	Multiply a matrix by a matrix.
	Basic code is used in many algorithms, mostly with minor changes such as scaling.
*/
void matrix_mul_matrix(ee_u32 N, MATRES C[N][N], MATDAT A[N][N], MATDAT B[N][N]) {
	ee_u32 i,j,k;
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			C[i][j]=0;
			for(k=0;k<N;k++)
			{
				C[i][j]+=A[i][k] * B[k][j];
			}
		}
	}
}

/* Function: matrix_mul_matrix_bitextract
	Multiply a matrix by a matrix, and extract some bits from the result.
	Basic code is used in many algorithms, mostly with minor changes such as scaling.
*/
void matrix_mul_matrix_bitextract(ee_u32 N, MATRES C[N][N], MATDAT A[N][N], MATDAT B[N][N]) {
	ee_u32 i,j,k;
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			C[i][j]=0;
			for(k=0;k<N;k++)
			{
				MATRES tmp=A[i][k] * B[k][j];
				C[i][j]+=bit_extract(tmp,2,4)*bit_extract(tmp,5,7);
			}
		}
	}
}
