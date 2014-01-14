// side effect in a subscript on rhs
// we want i=1 at the end

int main()
{
  int i=0, j=0;
  int a[10];
  
  j = a[i++];
  
  return 0;
}

