
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <fftw3.h>
#include "varglob.h"
#include "io.h"


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

int getopt_long(int argc, char * const argv[],
                  const char *optstring,
                  const struct option *longopts, int *longindex) {
	return argc;
}


#ifdef _GRAPHICS_
void graphic_destroy(void) {
  printf("");
}
void graphic_draw(int argc, char **argv, int histo[NCELL][NCELL][NCELL]) {
  printf("");
}
#endif

#ifdef _GLGRAPHICS_
void graphic_gldestroy(void) {
  printf("");
}
void graphic_glupdate(coord pos[NCELL][NCELL][NCELL]) {
  int i,j,k;
  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
        float x = pos[i][j][k]._[0]-3;
        float y = pos[i][j][k]._[1]-3;
        float z = pos[i][j][k]._[2]-3;
        printf("%f %f %f\n",x,y,z);
      }
    }
  }
}
void graphic_gldraw(int argc, char **argv, coord pos_[NCELL][NCELL][NCELL]) {
  printf("");
}
#endif
