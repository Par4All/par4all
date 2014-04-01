/* array k must be explicitly sized if the initialization is split.
 *
 * Simplified version of split_initializations02
 */

void split_initializations12(void)
{
  int k[][2] = { {1, 0}, {2, 0}, {3, 0} };
}
