/* Make sure that side effects in expressions used in array
   declarations are taken into account.

   This is not a common feature, but Pierre Villalon has spotted a
   function call used in an array declaration of some real code. And
   from a call to a side effect, there is not much.

   A bit more complicated than declaration02.c: side effect with
   function call in dimension declaration.
 */

int foo(int i)
{
  return i + 20;
}

int main (int argc, char** argv) {
  int i = 1;
  int j;
  long long size[i++][j=foo(i)];
  return (int) size[0][0];
}
