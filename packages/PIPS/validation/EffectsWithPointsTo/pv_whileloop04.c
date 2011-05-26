/* while loop case: pointers assigned inside loop */
int main()
{
  int i=0;
  int *p[10];
  int a[10];

  while(i<10)
    {
      p[i] = & a[i];
      i++;
    }

  return(0);
}
