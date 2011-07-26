// to test chains for array elements in a simple block
// as array references.
// with simple effects, we find out that there is no dependence between the statements
// defining a[1] and a[2]; but with convex effects, the information is hidden
// in the predicates, and we assume a dependence.

int main()
{
  int a[10];
  a[1] = 1;
  a[2] = 2;
  return a[1]+a[2];
}
