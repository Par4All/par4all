/* Check intermediate returns */

int for_loop09()
{
  int i;
  for(i=0;i!=5;i++) {
    if (i == 3) {
      printf("%d",i);
      return i;
    }
  }
  printf("Exit with %d",i);
  return i;
}
