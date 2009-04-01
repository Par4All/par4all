int main () {
  float a[10][10][10][10][10];
  float b[10][10][10][10][10];
  int i,j,k = 17,l,m = 120;
  float x = 2.12;
  float w = 1.0;

  for (i = 0; i < 10; i++) {
    for (j = 2; j < 7; j++) {
      for (k = 4; k < 3; k++) {
	for (l = 2; l < 10; l++) {
	  for (m = 0; m < 10; m++) {
	    float y;
	    float p = 2.369;
	    float z=2.32 + y;
	    y = 3.5 + x;	    
	    w=1.2 + z + p;
	    a[i][j][k][l][m] = x*y+x+z;
	  }
	}
      }
    }
  }
  for (m = 0; m < 10; m++) {
    float y;
    float p = 2.369;
    float z=2.32 + y;
    y = 3.5 + x;	    
    w=1.2 + z + p;
    a[1][1][1][1][m] = x*y+x+z;
  }
  return 0;
}
