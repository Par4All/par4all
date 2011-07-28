// to test chains for scalars in a simple block
// there must be no dependence between "a=0;" and "b=a" or "b=a+1".
// an there must be no dependence between "b=a" and the return statement. 

int main()
{
  int a,b;
  a = 0;
  a = 1;
  b = a;
  b = a+1;
  return b;
}
