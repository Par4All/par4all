/* Derived from rsp05.c to debug the parser when comments are used */
/* #include <stdio.h> */


/* Comment of a function declaration */
int filter();

/* Comment of a function body*/
int  main()
{
  /* Declaration comment */
  int i;
  int j;

  /* Function call comment*/
  j = filter(i);
  i = j;
  return 0;
}

int  filter(int x)
{
  int y;
  int res;

  /* Assignment comment */
  y = x;

  /* Return comment */
  return res;
  y = 0;
}
