/* Check  for constant first argument of modulo */

/* Issue: the constant value of the parameter a, 4, is not exploited
   when transformers are refined */

/* Second issue: the value of a, when set explictly in the caller is
   not used to compute the transformer test */

#include <stdio.h>

void check_modulo(int a)
{
  int b, r;

  a = 4;

  if(b>3 && b<10) {
    r = a % b;
    printf("a=%d, b=%d, r=%d\n", a, b, r);
    ;
  }
  else if (-10<b && b<-3) {
    r = a % b;
    printf("a=%d, b=%d, r=%d\n", a, b, r);
  }
  else if(-10<b && b<10) {
    r = a % b;
    printf("a=%d, b=%d, r=%d\n", a, b, r);
  }
  else if(b>5) {
    r = a % b;
    printf("a=%d, b=%d, r=%d\n", a, b, r);
  }
  else if(b<-5) {
    r = a % b;
    printf("a=%d, b=%d, r=%d\n", a, b, r);
  }
  else {
    r = a % b;
    printf("a=%d, b=%d, r=%d\n", a, b, r);
  }
}

int main()
{
  check_modulo(4);
  //check_modulo(0);
  //check_modulo(-4);

  return 0;
}
