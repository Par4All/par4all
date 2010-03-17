main()
{
  int i = 1;
  if (i==1) goto lab1;
  if (i==2) goto lab2;
  goto labdefault;
 lab1: printf("1 = %d",i);
  goto switch_exit;
 lab2: printf("2 = %d",i);
  goto switch_exit;
 labdefault: ;
 switch_exit:;
}
