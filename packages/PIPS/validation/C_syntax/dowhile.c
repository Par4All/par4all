main()
{
  int c = 5; /* declaration comment that should be lost by parser */
  /* Beginning of do while */
  do
    {
      --c;
      c++;
      c -= 1;
    } /* End of do while */
  while (c>0);
  /* End of function */
  return 0;
}
