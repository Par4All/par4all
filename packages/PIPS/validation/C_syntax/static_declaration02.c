// Static and int can be interchange, but PIPS prettyprinted knows
// only static int

// All variables declared together have the same storage: z is static

int main()
{
  static int x = 2;
  int static y = 3, z = 0;
  z++;
  x++;
  return y;
}
