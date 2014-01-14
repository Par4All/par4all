// side effect in a subscript on lhs
// we want i=1 at the end
// the effects is strange 
//             <    is written>: a[i++] i

int main()
{
  int i=0;
  int a[10];
  
  a[i++] = 0;
  
  return 0;
}

