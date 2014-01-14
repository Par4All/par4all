
int main()
{
  int i, a[10], b[10], c, d;
  
  for(i=0; i<10; i++) {
    a[i] = i;
    c=i;
  }
  
  //if there is a mult with more than 1 variable in rhs  transformer lose precision.
  for(i=0; i<10; i++) {
    b[i] = i*i;
    d=i*i;
  }
  
  return 0;
}
