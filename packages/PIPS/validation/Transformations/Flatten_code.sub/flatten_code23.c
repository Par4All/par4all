/* Basic test case: make sure no useless renaming occurs for variables
 */

int flatten_code23()
{
  int i = 1;
  {
    int i = 2;
    int j = 3;
  }
}
