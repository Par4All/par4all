/* Check that a static initialization does not generate a write effect
   on the static variable. The static initialization is not executed
   when the function is called. It is executed statically at compiler
   and/or link time, before the execution starts.

   Also, the information for anc initialization should be a
   MUST not a MAY in my opinion. But since there is no effect at all...
 */

void static02()
{
  static int i = 0;
}

main()
{
  static02();
}
