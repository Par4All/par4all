int temp; 
int
main (void) 
{
  int i,j,c,a[100]; 
  c = 2; 
  for (i = 0;i<100;i++) 
    {
      a[i] = c*a[i]+(a[i]-1); 
    } 
}
