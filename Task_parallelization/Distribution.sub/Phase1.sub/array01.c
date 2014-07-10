//init a and first column of b
//init b and all cluster use the first column for that
//transform 2D array (b) into 1D array (c)

int main() {
  int i;
  int a[4];
  int b[4][5];
  
#pragma distributed on_cluster=0
  for(i=0; i<4; i++) {
    a[i] = i;
    b[i][0] = a[i];
  }
  
#pragma distributed on_cluster=0
  {
    int j;
    for(j=1; j<5; j++) {
      b[0][j] = b[0][0] + j*10;
    }
  }
#pragma distributed on_cluster=1
  {
    int j;
    for(j=1; j<5; j++) {
      b[1][j] = b[1][0] + j*10;
    }
  }
#pragma distributed on_cluster=2
  {
    int j;
    for(j=1; j<5; j++) {
      b[2][j] = b[2][0] + j*10;
    }
  }
#pragma distributed on_cluster=3
  {
    int j;
    for(j=1; j<5; j++) {
      b[3][j] = b[3][0] + j*10;
    }
  }
  
  int c[20];
#pragma distributed on_cluster=0
  {
    int j, k;
    for(i=0; i<4; i++) {
      for(j=1; j<5; j++) {
        k = i*4+j;
        c[k] = b[i][j];
      }
    }
  }
  
  return 0;
}
