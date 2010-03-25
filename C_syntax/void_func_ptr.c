/* Bug in handling the functional pointer, probably interpreted as a
   function. No fix but work around the bug in util.c,
   RemoveFromExterns() */
void void_func_ptr(void (*func)(void *))
{
}
