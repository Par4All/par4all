/* while loop case: scalar pointer assigned inside loop */
int main()
{
  int i=0;
  int *p;
  int a[10];

  while(i<10)
    {
      p = & a[i];
      i++;
    }

  return(0);
}
