/* Check that parallel loops located inside a non-convertibal for foop are parallelized */


void for02(int n)
{
  int j;
  float x[100];
  float t, delta_t, t_max;

  for(t = 0.0; t<t_max; t = t + delta_t) {
    for(j=0;j<100;j++) {
      x[j] = 0.;
    }
  }
}
