#ifdef P4A_ACCEL_OPENMP




#elif defined(P4A_ACCEL_CUDA)
/**
 * Convert calls to fftw library to cufft using a runtime
 * It implies using --com-optimization runtime
 */


#include <cufft.h>

extern "C" {
#include <stdlib.h>
#include <p4a_accel.h>

typedef float fftwf_complex[2];

typedef struct fftwf_plan_s {
  cufftHandle cu_plan;
  int x,y,z;
  cufftComplex *in;
  cufftComplex *out;
  int direction;
  int params;
  size_t datasize;
} fftwf_plan;

fftwf_plan *fftwf_plan_dft_3d(int x,
                              int y,
                              int z,
                              fftwf_complex *in,
                              fftwf_complex *out,
                              int direction,
                              int params) {
  fftwf_plan *plan = (fftwf_plan *)malloc(sizeof(fftwf_plan));

  plan->cu_plan= NULL;
  plan->x = x;
  plan->y = y;
  plan->z = z;
  plan->in = (cufftComplex *)in;
  plan->out = (cufftComplex *)out;
  plan->direction = direction;
  plan->params = params;
  plan->datasize = x*y*z*sizeof(cufftComplex);

  cufftPlan3d(&(plan->cu_plan),x,y,z,CUFFT_C2C);

  return plan;
}

void fftwf_destroy_plan(fftwf_plan *plan) {
  cufftDestroy(plan->cu_plan);
//  *plan = { NULL, 0,0,0,NULL,NULL,0,0};
  free(plan);
}


// FFTW3 constant
#define FFTW_FORWARD (-1)
#define FFTW_BACKWARD (+1)

void fftwf_execute(fftwf_plan *plan) {
#ifdef P4A_RUNTIME_DEBUG
    fprintf(stderr,"Running fftwf_execute wrapper in P4A cuda, direction is : %d ",plan->direction);
#endif

    cufftComplex *in,*out;
#ifndef P4A_COMMUNICATION_RUNTIME
    P4A_accel_malloc((void **)&in, plan->datasize);
    P4A_accel_malloc((void **)&out, plan->datasize);
    P4A_copy_to_accel(plan->datasize,plan->in,in);
#else
    in  = (cufftComplex *)P4A_runtime_host_ptr_to_accel_ptr(plan->in,plan->datasize);
    out = (cufftComplex *)P4A_runtime_host_ptr_to_accel_ptr(plan->out,plan->datasize);
#endif


  if(plan->direction==FFTW_FORWARD) {
    cufftExecC2C(plan->cu_plan,
                 in,
                 out,
                 CUFFT_FORWARD);
  } else if(plan->direction==FFTW_BACKWARD ) {
    cufftExecC2C(plan->cu_plan,
                 in,
                 out,
                 CUFFT_INVERSE);
  } else {
    fprintf(stderr,"[%s:%d] Error : unknown direction (%d)\n",__FUNCTION__,__LINE__,plan->direction);
    exit(-1);
  }

#ifndef P4A_COMMUNICATION_RUNTIME
    P4A_copy_from_accel(plan->datasize,plan->in,in);
    P4A_accel_free(in);
    P4A_accel_free(out);
#endif
}



} // Extern "C"

#else
#error "You have to define either P4A_ACCEL_CUDA or P4A_ACCEL_OPENMP"
#endif
