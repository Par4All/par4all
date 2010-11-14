// See how C array declarations are exploited by semantics...

int trust01(int n)
{
  float a[n];
  // Trusting the following declarations makes the function inexecutable...
  // float b[0]; if array declarations are trusted in C as in Fortran

  // Not possible: float c[-1];

  int i = n;
  {
    int m;
    float b[m];
    i = i + 1;
  }
  return i;
}
