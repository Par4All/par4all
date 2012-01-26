#include <stdio.h>

void write_array(
		int 		nSizeWidth,
		int 		nSizeHeight,
		float 		dest[nSizeHeight][nSizeWidth],
		float 		source[nSizeHeight][nSizeWidth]
		)
{
   int y, x;


  // loop unroll for source == dest (further more, both images have padding, beware of indices
  for (y = 0; y < nSizeHeight; ++y) {
    for (x = 0; x < nSizeWidth; ++x)
      dest[y][x] = source[y][x]; // !! padding source

  }
}

/* ----------------------------------------------------------------------- */



int main() {
  int nSizeWidth=100; 
  int nSizeHeight=100;
  
  float src[nSizeHeight][nSizeWidth];
  float dest[nSizeHeight][nSizeWidth];
 
  
  write_array(nSizeWidth,nSizeHeight,
            dest,
            src);
  
  write_array(nSizeWidth,nSizeHeight,
            *((float(*)[nSizeHeight][nSizeWidth])dest[0]),
            *((float(*)[nSizeHeight][nSizeWidth])src[0]));

  return 0;
}
