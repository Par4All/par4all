// to test chains for array elements in a simple block
// we should find out that there is no dependence between "a[1]=1"
// and the return statement;
// however, the kill test is not yet sufficiently precise for
// simple effects, and the information is hidden in the predicate
// for convex effects

int main()
{
  int a[10];
  a[1] = 1;
  a[1] = 2;
  return a[1];
}
