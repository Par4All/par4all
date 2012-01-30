/* effects of non-convertible for loop on loop increment expression */


void for02(int n)
{
  int j;
  float x[100];
  float t, delta_t, t_max;

  for(t = 0.0; t<t_max; t += delta_t) {
    for(j=0;j<100;j++) {
      x[j] = 0.;
    }
  }
}
