// to test chains for arrays in a simple block
// as array references (here a[i] are functons of the store
// there must be dependences between sthe statements
// using/defining a[i]

int main()
{
  int a[10],i;
  i = 0;
  a[i] = 1;
  i = 1;
  a[i] = 2;
  i = 3;
  return a[i];
}
