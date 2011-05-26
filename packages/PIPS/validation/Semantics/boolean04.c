// Check boolean binary operators

//typedef enum { false, true } bool;

#include <stdbool.h>

int main(void)
{
  bool b1, b2, b_or, b_and, b_equal, b_non_equal, b_xor;

  // b1==b2==0: expect b_and==0, b_equal==1, b_non_equal==0, b_or==0
  b1 = b2 = false;
  b_or=b1||b2, b_and=b1&&b2, b_equal=b1==b2, b_non_equal=b1!=b2, b_xor=b1^b2;

  // b1==1, b2==0: expect b_and==0, b_equal==0, b_non_equal==1,b_or==1
  b1 = true, b2 = false;
  b_or = b1||b2, b_and = b1&&b2, b_equal=b1==b2, b_non_equal= b1!=b2, b_xor=b1^b2;

  // b1==0, b2==1: expect b_and==0, b_equal==0, b_non_equal==1, b_or==1
  b1 = false, b2 = true;
  b_or = b1||b2, b_and = b1&&b2, b_equal=b1==b2, b_non_equal= b1!=b2, b_xor=b1^b2;

  // b1==b2==1: expect b_and==1, b_equal==1, b_non_equal==0, b_or==1
  b1 = b2 = true;
  b_or = b1||b2, b_and = b1&&b2, b_equal=b1==b2, b_non_equal= b1!=b2, b_xor=b1^b2;

  return 0;
}
