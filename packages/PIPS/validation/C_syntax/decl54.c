/* Simplified version of decl52.c to understand a bug: the qualifier
   "const" is copied from the ycontext for i into the ycontext of
   j. So j ends up with a "const" qualifier like i. */

typedef struct _hdl_progress_handler
{
  const int i;
  int j;
} HDL_progress_handler;
