/* Stack management for PIPS contexts
 *
 * A PIPS context contains a pointer to a long jump buffer and a pointer in the
 * PIPS debug level stack. The long jump buffers are initialized by the
 * caller since set_jmp() must be performed by the procedure catching
 * the exception. This package only stores pointers towards these buffers.
 *
 * When a PIPS context is popped, the debug level stack is restored.
 *
 * The PIPS context stack pointer, pcsp, points to the top of the stack.
 * The empty stack is represented by pcsp == -1.
 *
 * Example: voir catch_user_error() in library pipmake, or a PIPS user
 * interface such as wpips, tpips or pips
 */

#include <stdio.h>
#include <setjmp.h>
	
#include "genC.h"
#include "misc.h"

typedef struct pips_context {
    jmp_buf * long_jump_buffer;
    int debug_stack_pointer;
} pips_context;

#define PIPS_CONTEXT_STACK_DEPTH 4
static pips_context pips_context_stack[PIPS_CONTEXT_STACK_DEPTH];
static int pcsp = -1;

jmp_buf * top_pips_context_stack()
{
    message_assert ("top_pips_context_stack",pcsp>=0)

    return pips_context_stack[pcsp].long_jump_buffer;
}


void set_pips_context_stack()
{
    /* pcsp == -1 when this function is called */
    if (pcsp!=-1) {
	pips_error("set_pips_context_stack",
		   "Unexpected pointer value %d\n", pcsp);
    }
}

void pop_pips_context()
{
    if(pcsp>=0) {
	set_debug_stack_pointer(pips_context_stack[pcsp].debug_stack_pointer);
	pcsp--;
    }
    else {
	/* Once PIPS is started, the stack should never be empty again...
	 but tpips is easier to program with an empty stack... */
	pips_error ("pop_pips_context",
		    "No more handler for user_error()!\n");
    }
}

void push_pips_context(jmp_buf * pljb)
{
    int dsp = get_debug_stack_pointer();

    pips_assert("push_pips_context", pcsp < PIPS_CONTEXT_STACK_DEPTH-1);

    pcsp++;
    pips_context_stack[pcsp].long_jump_buffer = pljb;
    pips_context_stack[pcsp].debug_stack_pointer = dsp;
}

void print_pips_context_stack()
{
    int i;

    (void) printf("Debug stack (last debug_on first): ");
    for(i=pcsp;i>=0;i--) {
	(void) printf("Debug stack pointer %d: debug level %d, long jump %p\n ", i,
		      pips_context_stack[i].debug_stack_pointer,
		      pips_context_stack[i].long_jump_buffer);
    }
}

/* That's all */
