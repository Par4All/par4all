main()
{
  int i = 0;
  if (i>1) 
    goto lab1;
  else 
    goto lab2;
 lab1: i = i+1;
 lab2: i = i-1;
  printf("%d\n",i);
}
