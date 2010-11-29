/* do while loop case: pointers assigned inside loop */
int main()
{
  int i=0;
  int *p[10];
  int a[10];

  do
    {
      p[i] = & a[i];
      i++;
    } while(i<10);

  return(0);
}
