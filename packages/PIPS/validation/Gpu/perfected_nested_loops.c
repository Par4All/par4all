int main() {
  int i,j;
  int Y = 100;
  int X = 100;
  int A[X][Y];
  int u1[X];
  int u2[X];
  int v1[Y];
  int v2[Y];
  
  for (i = 0; i < Y; i++) {
    register int u1_0 = v1[i];
    register int u2_0 = v2[i];
    for (j = 0; j < Y; j++) {
      A[i][j] = A[i][j] + u1_0 * v1[j] + u2_0 * v2[j];
    }
  }
}


