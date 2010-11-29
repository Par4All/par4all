/* loop case: no pointer is modified inside loop */
int main()
{
  int i;
  int a[10];

  for(i = 0; i<10; i++)
    a[i] = i;

  return(0);
}
