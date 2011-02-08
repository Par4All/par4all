/* Test some deep comments in for loops */
int main()
{
  int i;
  i = 0;
  /* This is an overcommented C99 for loop :-) */
  for(/* before init */int j = 0/* after init */; /* before cond */j < 5/* after cond */; /* before inc */j++/* after inc */) {
    /* loop body */
    i++;
    /* end of loop body */
  }
  /* after the loop */
  return 0;
}
