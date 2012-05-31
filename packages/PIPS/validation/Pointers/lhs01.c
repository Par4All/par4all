/* #include <stdio.h> */

void lhs01()
{
  int i = 2;
  int j = 2;
  int k = 2;

  /* This is a deprecated construct which is accepted by PIPS parser */
  //i>2?i:j=3;
  //(i>2?i:j)=3;

  *(i>2?&i:&j) = 3;
  /* printf("i=%d, j=%d\n", i, j); */

  /* This is a deprecated construct which is accepted by PIPS parser */
  //i>2?i=3:j=3;
  /* This is OK*/
  i = j = 2;
  i>2?(i=3):(j=3);

  i = j = 2;
  if(i>2)
    i = 3;
  else
    j = 3;

  i = j = 2;
  i>2?(j=4):(j=3);

  if(i>2)
    j = 4;
  else
    j = 3;
}

int main()
{
  lhs01();
  return 0;
}
