// here, the loop index must not appear in the live in paths of the loop
// but must appear in the live out paths of the loop body because it may
// be used afterwards (in the return statement)

int main()
{
  int i, a, b;

  a = 0;
  b = 1;
  for(i=0; i<10; i++)
    a = a+b;

  return a+i;
}
