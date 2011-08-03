// Make sure that parentheses are used when requested by property
// PRETTYPRINT_ALL_PARENTHESES

int main () {
  float a, b, c, d, e;
  d = (a + c) - b;
  d = a + (c - b);
  return (int) d;
}
