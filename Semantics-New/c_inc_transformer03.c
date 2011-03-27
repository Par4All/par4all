/* modified version of c_inc_transformer02.c */


int main(int j, char **unused) {
  double A[10];
  int B[10];
  int k = 2;

  A[j++] = 0.;
  B[j++] = 0;
  B[1] = k++;
  A[1] = (double) k++;
  return j;
}
