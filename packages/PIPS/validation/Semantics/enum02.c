/* Check compatibility of enum and int */

typedef enum { false, true } bool;

int main(void)
{
  int b;
  b = true;

  b = true + 1;

  b += true;

  return b;
}
