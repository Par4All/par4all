/* 
 * $Id$
 */

int loop_statistics(string name)
{
  debug_on("STATISTICS_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", name);

  /* ... */

  pips_debug(1, "done.\n");
  debug_off();
  return TRUE;
}
