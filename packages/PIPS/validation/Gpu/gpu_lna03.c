
/*
 * file for p4a_kernel_launcher_1.c
 */
typedef float float_t;
extern void p4a_kernel_wrapper_1(float_t save[501][501], float_t space[501][501], int i, int j);
extern void p4a_kernel_wrapper_2(float_t save[501][501], float_t space[501][501], float_t dest[501][501], int i, int j, int k);

void p4a_kernel_launcher_1(float_t save[501][501], float_t space[501][501])
{
   int i;
   int j;
   int k;
   float_t dest[501][501];

kernel2:
   for(i = 0; i <= 123; i += 1)
      for(j = 0; j <= 498; j += 1)
         p4a_kernel_wrapper_1(save, space, i+1, j+1);

kernel3:
   for(i = 0; i <= 123; i += 1)
      for(j = 0; j <= 456; j += 1)
	for(k = 0; k <= 789; k += 1)
	  p4a_kernel_wrapper_2(save, space, dest, i+1, j+1, k+1);

}

