/* while loop case: no pointer is modified inside loop */
int main()
{
  int i=0;
  int a[10];

  while(i<10)
    {
      a[i] = i;
      i++;
    }

  return(0);
}
