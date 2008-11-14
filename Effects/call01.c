/* #include<stdio.h> */

void call01(int * pi)
{
  *pi = 1;
}

main()
{
  int i;
  call01(&i);
}
