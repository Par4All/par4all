/* Effect bug in ammp: call site to a_m_serial seems to lack argument(s) */
/* implicit typing for activate seems to lead to a bug for the
   substitution of the dummy parameters by formal parameters. Also
   formal parameters can be defined in different ways. Here we use
   the old style version of ammp01.

   See also old_decl01.c
 */

activate (i1, i2)
int i1;
int i2;
{
}
