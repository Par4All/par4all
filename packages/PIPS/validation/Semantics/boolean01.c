#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

bool alea(void)
{
  // rand() >=0
  // 0 <= ((>=0) % 2) <= 1
  return rand()%2;
}

int main(void)
{
  bool b1, b2, b3, b4, b5, b6, b7;
  b1 = true;
  b2 = false;
  // b3 in 0..1 because of type? %2?
  b3 = alea();
  // b4 in 0..1 because logical
  b4 = b1 && b3;
  // b5 in 0..1 because logical
  b5 = b1 || b3;
  // b6 in 0..1 because logical
  b6 = !b3;
  // b7: xor is a bitwise operator returning an int
  // it requires a special handling when both operand are boolean,
  // which is not (yet) implemented
  b7 = b1 ^ b2;
  return 0;
}
