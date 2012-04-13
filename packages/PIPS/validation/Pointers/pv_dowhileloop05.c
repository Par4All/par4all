/* do while loop case with pointer arithmetic: conservatively handled */
int main()
{
  int i=0;
  int *p, *q;

  p = (int *) malloc(10* sizeof(int));
  q = p;

  do
    {
      *q = i;
      q++;
      i++;
    } while(i<5);

  free(p);

  return(0);
}
