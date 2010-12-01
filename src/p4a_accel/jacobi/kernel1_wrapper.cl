typedef float float_t;
#define SIZE 501

inline void kernel1(__global float_t space[SIZE][SIZE],__global float_t save[SIZE][SIZE],int i, int j) 
{ 
  save[i][j] = 0.25*(space[i-1][j]+space[i+1][j]+space[i][j-1]+space[i][j+1]);
}

__kernel void kernel1_wrapper(__global float_t space[SIZE][SIZE], __global float_t save[SIZE][SIZE])
{
    
  int i = get_global_id(0);
  int j = get_global_id(1);
  if(i >= 1 && i <= SIZE - 2 && j >= 1 && j <= SIZE - 2)
    kernel1(space, save, i, j);
}
