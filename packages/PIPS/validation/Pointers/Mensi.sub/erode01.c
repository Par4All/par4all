// from /Paws/pyps_modules/paws_regions.c
#ifndef __PIPS__
#define MIN(a,b) ((a)<(b))?(a):(b)
#endif
#include <stdio.h>
#define kernel_size 5

void erode(int isi, int isj, int **new_image, int **image)
{

  int i, j, k,l;

  for(l=0;l<5;l++)

  loop1:   for(i =  kernel_size/2; i<isi - kernel_size/2; i++) {
      for(j =  kernel_size/2; j<isj - kernel_size/2; j++) {
	int l=image[i][j];
	for(k=0;k<kernel_size;k++) {
	  l = MIN(l,image[i][j+1-kernel_size/2+k]);
	}
	new_image[i][j] = l;
      }
    }
}


