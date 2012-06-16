// Pointer initializations to a 1-D array

int main()
{
  int a[10];
  int * p = a;
  int (*q)[10] = &a;
  int * r = &a[0];

  return 0;
}
