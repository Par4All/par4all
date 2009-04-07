int main(int argc, char **argv)
{
  int d, i, n=100;
  int x[n], y[n], t[n];

  for (i=0 ; i<n ; i++) {
    t[i] = x[i];
    x[i] = y[i];
    y[i] = t[i];
  }
}
