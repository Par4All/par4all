/* handling of cast: incompatible types
 *
 * src will point to an anywhere
 */

int main()
{
  float a = 1.0;
  unsigned char *src = (unsigned char *) &a;

  return 0;
}
