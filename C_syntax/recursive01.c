/* Make sure that the C parser can parse recursive calls.
 *
 * The bug was due to the return value, which hid the function itslef.
 */

int recursive01(int i)
{
  if(i>0)
    i = recursive01(i-1);
  return i;
}
