/* loop case: no pointer is modified inside loop */
int main()
{
  int i;
  int *p;

  p = (int *) malloc(10* sizeof(int));

  for(i = 0; i<10; i++)
    p[i] = i;

  return(0);
}
