// test case for GPU_LOOP_NEST_ANNOTATE
// when GPU_LOOP_NEST_ANNOTATE_PARALLEL is TRUE


int main(){
  float a[501][501][501][501];
  int i,j,k,l;
  int i2,j2,k2,l2;

  for(i = 0; i <= 498; i += 1)
    for(j = 0; j <= 498; j += 1)
      for(k = 0; k <= 498; k += 1)
	for(l = 0; l <= 498; l += 1)
	  a[i][j][k][l] = (float) i*j;

  for(i = 0; i <= 498; i += 1) {
    int i1 = 7 + i;
    for(j = 0; j <= 498; j += 1) {
      int j1 = j+5;
      for(k = 0; k <= 498; k += 1) {
	int k1 =3*k;
	for(l = 0; l <= 498; l += 1) {
	  int l1 = 2*l;
	  a[i][j][k][l] = i1 + j1 + k1+l1;
	}
      }
    }
  }

  for(i = 0; i <= 498; i += 1) {
    i2 = 7 + i;
    for(j = 0; j <= 498; j += 1) {
      j2 = j+5;
      for(k = 0; k <= 498; k += 1) {
	k2 =3*k;
	for(l = 0; l <= 498; l += 1) {
	  l2 = 2*l;
	  a[i][j][k][l] = i2 + j2 + k2 + l2;
	}
      }
    }
  }
  return 0;
}
