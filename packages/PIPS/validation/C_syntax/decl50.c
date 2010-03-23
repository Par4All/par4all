/* Simplified version of decl24. Bug with const and volatile
   declaration */

/*
typedef struct _hdl_progress_handler
{
  const void * display_data;
} HDL_progress_handler;
*/

/* pointer to a constant */

/* OK for int */
//const int * next_data;

/* But lost for void... The internal representation
   cannot keep the information. */
const void * next_data;
volatile void * previous_data;

// This is ok:
//void * const ultimate_data;
