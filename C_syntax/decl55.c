/* Simplified version of decl52.c to understand a bug: const is
   applied to the pointer, not to the int */

typedef struct _hdl_progress_handler
{
  const int * i;
  //int j;
} HDL_progress_handler;
