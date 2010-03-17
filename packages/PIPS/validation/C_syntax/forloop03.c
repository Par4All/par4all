/* Same as forloop02, but for C operators */

main()
{
  int i, j;
  int a[5];
  for (i=1; i<5; i = i + 1)
    {
      a[i] = i;
      /* Problem with simple effects in SearchIoElement: FMT= expected but not found */
      /* printf("%d\t%d\n",i,j); */
    }
}
