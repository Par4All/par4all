/* Basic test case: make sure no useless renaming occurs for typedef
 */

int flatten_code22()
{
  {
    typedef int myint1;
  }
  {
    typedef int myint1;
  }
}
