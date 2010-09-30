const int size = 100;

int main () {
  int i,j,k;
  int a[size][size][size];
 l300:
  for (i=0; i<size ;i++) {
    for (j=0; j<size ;j++) {
      for (k=0; k<size ;k++) {
	a[i][j][k] = 2;
      }
    }
  }
  return 0;
}
