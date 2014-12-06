/* Basic test case: make sure no useless renaming occurs for typedef
 */

#include <stdio.h>

int flatten_code20()
{
  typedef int myint1;
  {
    typedef int myint2;
  }
}
