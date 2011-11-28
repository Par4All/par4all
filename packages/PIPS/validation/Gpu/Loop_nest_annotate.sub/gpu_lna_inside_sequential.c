
/*
 * file for p4a_kernel_launcher_1.c
 */
typedef float float_t;
extern void p4a_kernel_wrapper_1(float_t save[501][501], float_t space[501][501], int i, int j);

void p4a_kernel_launcher_1(float_t save[501][501], float_t space[501][501])
{
   int seq;
   int i;
   int j;
   for(seq = 0; seq <= 2; seq += 1)
     for(i = 0; i <= 123; i += 1)
        for(j = 0; j <= 498; j += 1)
          save[i][j]=0;
}

