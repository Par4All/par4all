/* Check that Allen&Kennedy finds parallel loops and does not
   distribute loops for declarations in loop bodies */

int main () {
  float a[10][10][10][10][10];
  int i,j,k,l,m;
  float x;
  x = 2.12;

  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
      // comments 2
      float z = 0.0;
      z = 0.0;
      for (k = 0; k < 10; k++) {
	z = k * 2.0;
	for (l = 0; l < 10; l++) {
	  for (m = 0; m < 10; m++) {
	    //comments 3
	    float y;
	    y = 2.0;
	    // comments 4
	    y = 3.5 + x + z;
	    a[i][j][k][l][m] = x*y;
	  }
	}
      }
    }
  }
  return 0;
}
