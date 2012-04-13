/* do while loop case: scalar pointer assigned inside loop */
int main()
{
  int i=0;
  int *p;
  int a[10];

  do
    {
      p = & a[i];
      i++;
    } while(i<10);

  return(0);
}
