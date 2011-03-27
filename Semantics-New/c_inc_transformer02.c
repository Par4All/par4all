/* simplified version of c_inc_transformer01.c */


int main(int j, char **unused) {
  int A[1];

  A[1] = j++;
  return j;
}
