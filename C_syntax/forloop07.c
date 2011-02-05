/* Test some deep comments and line numbers in for loops */
int main()
{
  int i, j;
  i = 0;
  /* This is an overcommented diluted for loop to test line numbering
     :-) */
  for
    (
     /* before init */
     j = 0
       /* after init */;
     /* before cond */
     j < 5/* after cond */;
     /* before inc */
     j++
       /* after inc */
     )
    /* between the clause and block */
    {
      /* loop body */
      i++;
      /* end of loop body */
    }
  /* after the loop */
  return 0;
}
