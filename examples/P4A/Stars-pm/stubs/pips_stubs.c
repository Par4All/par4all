
#include <stdio.h>
#include <string.h>
#include "stars-pm.h"

#ifndef P4A_RUNTIME_FFTW
#include <fftw3.h>
void fftwf_free(void *p) {
  free(p);
}

void fftwf_destroy_plan( void *p) {
  free(p);
}


void fftwf_execute(fftwf_plan p) {
  printf("%p",p);
}

fftwf_plan fftwf_plan_dft_3d(int nx, int ny, int nz, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags) {
  fftwf_plan fft;
  return fft;
}

void* fftwf_malloc(int size) {
 return malloc(size);
}

void fftwf_init_threads() {
  fprintf(stderr,"FFTW STUB : %s\n",__FUNCTION__);
}
void fftwf_plan_with_nthreads(int nthreads) {
  fprintf(stderr,"FFTW STUB : %s\n",__FUNCTION__);
}
int omp_get_max_threads() {
  return rand();
}
#endif // P4A_RUNTIME_FFTW

int getopt_long(int argc, char * const argv[],
                  const char *optstring,
                  const struct option *longopts, int *longindex) {
	return argc;
}


#ifdef _GRAPHICS_
void graphic_destroy(void) {
  printf("");
}
void graphic_draw(int argc, char **argv, int histo[NP][NP][NP]) {
  printf("");
}
#endif

#ifdef _GLGRAPHICS_
void graphic_gldestroy(void) {
  printf("");
}
void graphic_glupdate(coord pos[NP][NP][NP]) {
  int i,j,k;
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        float x = pos[i][j][k]._[0]-3;
        float y = pos[i][j][k]._[1]-3;
        float z = pos[i][j][k]._[2]-3;
        printf("%f %f %f\n",x,y,z);
      }
    }
  }
}
void graphic_gldraw(int argc, char **argv, coord pos_[NP][NP][NP]) {
  printf("");
}
#endif
