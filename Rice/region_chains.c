int main () {
  float a[10][10][10][10][10];
  float b[10][10][10][10][10];
  int i,j,k = 17,l,m = 120;
  float x = 2.12;
  float w = 1.0;
  float y;

  for (i = 0; i < 10; i++) {
    for (j = 2; j < 7; j++) {
      for (k = 4; k < 10; k++) {
	for (l = 2; l < 10; l++) {
	  for (m = 0; m < 10; m++) {
	    y = 3.5 + x;	    
	    a[i][j][k][l][m] = x*y;
	  }
	}
      }
    }
  }
  for (m = 0; m < 10; m++) {
    y = 3.5 + x;	    
    a[i][j][k][l][m] = x*y;

  }
  return 0;
}
