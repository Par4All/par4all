typedef float float_t;
#define SIZE 501
#define P4A_accel_kernel inline void
#define P4A_accel_kernel_wrapper __kernel void
#define P4A_accel_global_address __global
#define P4A_vp_0 get_global_id(0)
#define P4A_vp_1 get_global_id(1)

// Un commentaire

/* Un commentaire en bloc
 */

P4A_accel_kernel kernel1(P4A_accel_global_address float_t space[SIZE][SIZE],P4A_accel_global_address float_t save[SIZE][SIZE],int i, int j) 
{ 
  save[i][j] = 0.25*(space[i-1][j]+space[i+1][j]+space[i][j-1]+space[i][j+1]);
}

P4A_accel_kernel_wrapper kernel1_wrapper(P4A_accel_global_address float_t space[SIZE][SIZE], P4A_accel_global_address float_t save[SIZE][SIZE])
{
    
  //int i = get_global_id(0);
  int i = P4A_vp_0;
  //int j = get_global_id(1);
  int j = P4A_vp_1;
  if(i >= 1 && i <= SIZE - 2 && j >= 1 && j <= SIZE - 2)
    kernel1(space, save, i, j);
}
