// Check the use and translation of anywhere effect

// Defined for a smaller version of pointer03.tpips, used for debugging

int x;
int y;
int z;

void foo(int *p)
{
  (*p)++;
}

int main()
{
  x= 1, y=2, z=3;
  int *p = (int *) malloc(sizeof(int));
  foo(p);
  p = &x;
  foo(p);
  return x+y+z; // no information left about x, y and z
}
