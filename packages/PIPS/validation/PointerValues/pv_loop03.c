/* loop case: scalar pointer assigned inside loop */
int main()
{
  int i;
  int *p;
  int a[10];

  for(i = 0; i<10; i++)
    p = & a[i];

  return(0);
}
