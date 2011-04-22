/* To check loop distribution. Same as loop_distribution_simple01.c,
   but the redundant initialization of z is removed. */

int main () {
  float a[10];
  int i;

  for (i = 0; i < 10; i++) {
    float z = 0.0;
    a[i] = z;
  }
  return 0;
}
