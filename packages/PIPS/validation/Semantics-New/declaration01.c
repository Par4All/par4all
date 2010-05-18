/* Make sure that side effects in expressions used in array
   declarations are taken into account.

   This is not a common feature, but Pierre Villalon has spotted a
   function call used in an array declaration of some real code. And
   from a call to a side effect, there is not much.
 */

int main (int argc, char** argv) {
  int i = 1;
  long long size[i++];
  return (int) size[0];
}
