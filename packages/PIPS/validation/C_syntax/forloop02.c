main()
{
  int i, j;
  int a[5];
  for (i=1, j=2; i<5; i++, j+=2)
    {
      a[i] = i;
      /* Problem with simple effects in SearchIoElement: FMT= expected but not found */
      /* printf("%d\t%d\n",i,j); */
    }
}
