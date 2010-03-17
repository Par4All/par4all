/* Same as forloop03, but to check for to do loop conversion */

main()
{
  int i, j;
  int a[5];
  for (i=1; i<5; i++)
    {
      a[i] = i;
      /* Problem with simple effects in SearchIoElement: FMT= expected but not found */
      /* printf("%d\t%d\n",i,j); */
    }
}
