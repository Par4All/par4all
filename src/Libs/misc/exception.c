/* Stack management for PIPS contexts
 *
 * A PIPS context contains a pointer in the PIPS debug level stack.
 * When a PIPS context is popped, the debug level stack is restored.
 * It may be the current level!
 *
 * Example: voir catch_user_error() in library pipmake, or a PIPS user
 * interface such as wpips, tpips or pips
 */

#include <stdio.h>
#include <genC.h>
#include "misc.h"

static stack debug_stack = NULL;

void push_pips_context(char * file, char * function, int line)
{
  pips_debug(9, "%s %s:%d\n", function, file, line);
  if (!debug_stack) debug_stack = stack_make(0, 50, 0);
  stack_push((void*) get_debug_stack_pointer(), debug_stack);
}

void pop_pips_context(char * file, char * function, int line)
{
  pips_debug(9, "%s %s:%d\n", function, file, line);
  if (!debug_stack) 
    pips_internal_error("unexpected pop without push %s %s:%d\n",
			function, file, line);
  set_debug_stack_pointer((int) stack_pop(debug_stack));
}

/* That's all */
