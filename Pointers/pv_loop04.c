/* loop case: pointers assigned inside loop */
int main()
{
  int i;
  int *p[10];
  int a[10];

  for(i = 0; i<10; i++)
    p[i] = & a[i];

  return(0);
}
