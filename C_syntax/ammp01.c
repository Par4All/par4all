/* Effect bug in ammp: call site to a_m_serial seems to lack argument(s) */
/* implicit typing for activate seems to lead to a bug for the
   substitution of the dummy parameters by formal parameters */

activate (int i1, int i2)
{
}
