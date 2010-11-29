/* do while loop case: no pointer is modified inside loop */
int main()
{
  int i=0;
  int a[10];

  do
    {
      a[i] = i;
      i++;
    } while(i<10);

  return(0);
}
