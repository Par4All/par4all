/* Basic test case: make sure no useless renaming occurs for variables
 *
 * No renaming is observed, with or without initializations
 */

#include <stdio.h>

int flatten_code19()
{
  int i = 1;
  {
    int j = 2;
    j = 3;
  }
  i = 2;
}
