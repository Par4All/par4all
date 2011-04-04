/* simpliffied version of c_inc_transformer03.c: check for casts */


int main(int j, char **unused) {
  double A[10];
  int k = 2;

  A[1] = (double) k++;
  return j+k;
}
