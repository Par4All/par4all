//
//	kernel_order_test.c
//    
//	Copyright 2011 HPC Project
//


#include <stdio.h> /* stderr */

#define BLOCKSIZE 16
#define MATSIZE 128

int main() {
    int i, k, j, cpi, cpj;		// indexes used in loops
    float l[MATSIZE*MATSIZE];
    
    // Number of blocks
    int n = MATSIZE / BLOCKSIZE;
    
    for (k = 0; k < n; k++) {
		
		/* Temporary block */
		float tmp[BLOCKSIZE * BLOCKSIZE];
		float _tmp1[BLOCKSIZE * BLOCKSIZE];
		
		for (cpi = 0; cpi < BLOCKSIZE * BLOCKSIZE; cpi++)
			tmp[cpi] = cpi;
		
		for (cpi = 0; cpi < BLOCKSIZE; cpi++)
			for (cpj = 0; cpj < BLOCKSIZE; cpj++)
				l[(k * BLOCKSIZE + cpi) * MATSIZE + k * BLOCKSIZE + cpj] =
					tmp[cpi * BLOCKSIZE + cpj];
		

	   for (cpi = 0; cpi < BLOCKSIZE; cpi++) {
				for (cpj = cpi+1; cpj < BLOCKSIZE; cpj++) {
					unsigned idx1 = cpi * BLOCKSIZE + cpj;
					unsigned idx2 = cpj * BLOCKSIZE + cpi;
					_tmp1[idx2] = tmp[idx1];
				}
			}
			
			for (cpi = 0; cpi < BLOCKSIZE; cpi++) {
				for (cpj = cpi + 1; cpj < BLOCKSIZE; cpj++) {
					unsigned idx1 = cpi * BLOCKSIZE + cpj;
					unsigned idx2 = cpj * BLOCKSIZE + cpi;
					tmp[idx1] = tmp[idx2];
					tmp[idx2] = _tmp1[idx2];
			}
		}
	
		for (i = k + 1; i < n; i++) {
			float lik[BLOCKSIZE * BLOCKSIZE];
			
			for (cpi = 0; cpi < BLOCKSIZE; cpi++)
				for (cpj = 0; cpj < BLOCKSIZE; cpj++)
					lik[cpi * BLOCKSIZE + cpj] = tmp[cpi * BLOCKSIZE + cpj] + k;
				
			for (cpi = 0; cpi < BLOCKSIZE; cpi++)
				for (cpj = 0; cpj < BLOCKSIZE; cpj++)
					l[(i * BLOCKSIZE + cpi) * MATSIZE + k * BLOCKSIZE + cpj] =
					lik[cpi * BLOCKSIZE + cpj];
		}

	
    }	// End of k loop 
      
 
       // Prints the result matrix l[0] in a file
    printf( "Block algorithm : L result matrix \n");
 
    for (i = 0; i < MATSIZE * MATSIZE; i++) {
		printf("% 6.3f ",  l[i]);
	} 
return 0;   
}

