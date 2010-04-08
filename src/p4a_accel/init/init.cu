#define P4A_DEBUG
#define P4A_ACCEL_CUDA
#include <p4a_accel.h>

/* Just some weird examples as a proof of concept of the P4A for CUDA API
*/

float execution_time;

#define N 16
#define M 256
float imagein_re[N][N], imageout_re[M][M];
float imagein_im[N][N], imageout_im[M][M];


P4A_accel_kernel_wrapper void pips_accel_1(float accel_imagein_re[N][N],
					   float accel_imagein_im[N][N]) {
  /* Get the virtual processor adresses */
  int i = P4A_vp_0;
  int j = P4A_vp_1;

  //  for(i=0;i<N;i++)
  // for(j=0;j<N;j++) {
  accel_imagein_re[i][j]=i+j/10000.0;
  accel_imagein_im[i][j]=j+i/10000.0;
  // }
}


P4A_accel_kernel_wrapper void pips_accel_2(float accel_imagein_re[N][N],
					   float accel_imagein_im[N][N]) {
  /* Get the virtual processor adresses */
  int i = P4A_vp_0;
  int j = P4A_vp_1;

  //  for(i=0;i<N;i++)
  // for(j=0;j<N;j++) {
      accel_imagein_re[i][j]= 0.5 + accel_imagein_im[i][j];
  // }
}


void getimage() {
  //int i,j;

  float (*accel_imagein_re)[N][N];
  float (*accel_imagein_im)[N][N];

  P4A_accel_malloc(&accel_imagein_re, sizeof(imagein_re));
  P4A_copy_to_accel(imagein_re, accel_imagein_re, sizeof(imagein_re));

  P4A_accel_malloc(&accel_imagein_im, sizeof(imagein_im));
  P4A_copy_to_accel(imagein_im, accel_imagein_im, sizeof(imagein_im));

  /*
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
    imagein_re[i][j]=0.0;
    imagein_im[i][j]=0.0;
    }
  */
  P4A_accel_timer_start;
  /* Call pips_accel_1 in a N*N parallel nested loop */
  P4A_call_accel_kernel_2d(pips_accel_1, N, N,
			   *accel_imagein_re, *accel_imagein_im);
  execution_time = P4A_accel_timer_stop_and_float_measure();
  fprintf(stderr, "Temps d'exécution : %f ms\n", execution_time);

  /*
    outline_compute:
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
    imagein_re[i][j]= 1 + imagein_im[i][j];
    }
  */
  dim3 pips_accel_dimBlock_2(N, N);
  P4A_accel_timer_start;
  P4A_call_accel_kernel((pips_accel_2, 1, pips_accel_dimBlock_2),
			(*accel_imagein_re, *accel_imagein_im));
  execution_time = P4A_accel_timer_stop_and_float_measure();
  fprintf(stderr, "Temps d'exécution : %f ms\n", execution_time);

  P4A_copy_from_accel(imagein_re, accel_imagein_re, sizeof(imagein_re));
  P4A_accel_free(accel_imagein_re);
  P4A_copy_from_accel(imagein_im, accel_imagein_im, sizeof(imagein_im));
  P4A_accel_free(accel_imagein_im);
  for( unsigned int i = 0; i < N; ++i) {
    printf("Line %d: ", i);
    for( unsigned int j = 0; j < N; ++j)
      printf("%f ", imagein_re[i][j]);
    puts("\n");
  }
}


/* For this kernel invocation: */
enum { num_threads = 32};

__global__ void
testKernel( float g_idata[num_threads], float g_odata[num_threads])
{
  // access thread id
  const unsigned int tid = P4A_vp_0;
  // access number of threads in this block
  //const unsigned int num_threads = blockDim.x;

  g_odata[tid] = tid*g_idata[tid]+blockIdx.x;
}

int main() {
  P4A_init_accel;

  //unsigned int num_threads = 32;
  unsigned int mem_size = sizeof( float) * num_threads;

  // allocate host memory
  float h_idata[mem_size];
  // initalize the memory
  for( unsigned int i = 0; i < num_threads; ++i)
    h_idata[i] = (float) i;

  // allocate device memory
  float (*d_idata)[mem_size];
  P4A_accel_malloc(&d_idata, sizeof(h_idata));
  // copy host memory to device
  P4A_copy_to_accel(h_idata, *d_idata, sizeof(h_idata));

  // allocate device memory for result
  float (*d_odata)[mem_size];
  cutilSafeCall( cudaMalloc( (void**) &d_odata, sizeof(h_idata)));

  // setup execution parameters
  dim3 grid( 1, 1, 1);
  dim3 threads( num_threads, 1, 1);

  // execute the kernel:
  P4A_call_accel_kernel((testKernel, grid, threads), (*d_idata, *d_odata));

  // The same but with a higher level call:
  P4A_call_accel_kernel_1d(testKernel, num_threads, *d_idata, *d_odata);

  // allocate mem for the result on host side
  float h_odata[mem_size];
  // copy result from device to host
  P4A_copy_from_accel(h_odata, d_odata, sizeof(h_odata));

  for( unsigned int i = 0; i < num_threads; ++i)
    printf("%f ", h_odata[i]);
  puts("\n");

  getimage();
  P4A_release_accel;
  return 0;
}
