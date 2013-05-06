/* A double dereferencing should generate two levels of stubs, _a1_1 and _a1_1_1
 *
 */

void double_pointer03(float**  a1)
{
  float y = **a1;
  **a1 = 0;
  return ;
}
