
/*
 * file for p4a_kernel_launcher_1.c
 */
typedef float float_t;
extern void p4a_kernel_wrapper_1(float_t save[501][501], float_t space[501][501], int i);

void p4a_kernel_launcher_1(float_t save[501][501], float_t space[501][501])
{
   int i;
kernel2:
   for(i = 0; i <= 498; i += 1)
     p4a_kernel_wrapper_1(save, space, i+1);
}

