//same that side_effect03 but with the property 
// setproperty SEMANTICS_ANALYZE_CONSTANT_PATH TRUE
// side effect in a subscript on lhs
// we want i=1 at the end
// if we have a[*]=0, it can be ok
// if we have a[0]=0, it will be perfect
// the effects is strange 
//             <    is written>: a[i++] i

int main()
{
  int i=0;
  int a[10];
  
  a[i++] = 0;
  
  return 0;
}

