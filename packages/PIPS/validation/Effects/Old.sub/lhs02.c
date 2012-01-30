/* To check the behavior of conditional expressions, especially in the lhs. */

void lhs02()
{
  int i = 2;
  int j = 2;

  *(i>2?&i:&j) = 3;

  j = i>2? i+1 : j+2;
}
