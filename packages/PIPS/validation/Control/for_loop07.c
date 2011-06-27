/* infinite loop in a for loop body*/

void foo()
{
  int i;
  for(i=0;i!=5;i++) {
  more:
    if (i == 3)
      printf("%d",i);
    goto more;
  }
  printf("Exit with %d",i);
}
