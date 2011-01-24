#include "stdio.h"

int main () {
  float a[10][10][10][10][10];
  int i,j,k,l,m;
  float x = 2.12;
  float z = 0.0;
  float y = 2.0;

  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
      for (k = 0; k < 10; k++) {
	z = k * 2.0;
	for (l = 0; l < 10; l++) {
	  for (m = 0; m < 10; m++) {
	    y = 3.5 + x + z;
	    a[i][j][k][l][m] = x*y;
	  }
	}
      }
    }
  }

  // use the value of the array to prevent pips doing optimization on unused
  // values
  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
      for (k = 0; k < 10; k++) {
	z = k * 2.0;
	for (l = 0; l < 10; l++) {
	  for (m = 0; m < 10; m++) {
	    fprintf (stdout, "%f", a[i][j][k][l][m]);
	  }
	}
      }
    }
  }

  return 0;
}
