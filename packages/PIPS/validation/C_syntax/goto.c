main()
{
  int i = 0;
  /* if with gotos */
  if (i>1) 
    /* goto lab1 */
    goto lab1;
  else 
    /* goto lab2 */
    goto lab2;
  /* lab1 statement */
 lab1: i = i+1;
  /* lab2 statement */
 lab2: i = i-1;
  printf("%d\n",i);
}
