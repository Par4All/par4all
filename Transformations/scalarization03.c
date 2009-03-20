int main(int argc, char **argv)
{
  int d, i, n=100;
  int x[n], y[n], t[n]; // arrays
  int *x2, *y2, *t2;    // pointers

  /* Expected result: t[i] should be scalarized */
  for (i=0 ; i<n ; i++) {
    t[i] = x[i];
    x[i] = y[i];
    y[i] = t[i];
  }

  /* Expected result: t2[i] should(?) be scalarized */
  for (i=0 ; i<n ; i++) {
    t2[i] = x2[i];
    x2[i] = y2[i];
    y2[i] = t2[i];
  }

  d = 1;
  return d;
}
