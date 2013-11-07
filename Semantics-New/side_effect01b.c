// make a side effect in the rhs
// we want i=1 at the end

int main()
{
  int i=0;
  int x= (i=1)+2;
  
  return 0;
}