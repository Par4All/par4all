/*********************************************************************************/
/* MEM_SPY : Package to track memory usage. - Beatrice Creusilllet - August 1995 */
/*********************************************************************************/

/* Usage:
 *
 *  ~ mem_spy_init() allocates and initializes all that is needed.
 *  ~ mem_spy_reset() frees the allocated memory after using mem_spy.
 *  ~ put mem_spy_begin() at the entry of the code section you want to
 *    spy, and mem_spy_end("<whatever you want>") at the end. It will
 *    print a line on stderr with the following format:
 * 
 *    MEM_SPY <whatever you want> begin: *MB end: *MB diff: *MB proper: *MB
 *   
 *    each * being a floating number. 
 *
 *       ~ <begin> is the memory usage at the entry of the code section.
 *       ~ <end> is the memory usage at the end of the code section.
 *       ~ <diff> is the memory allocation between the two points.
 *       ~ <proper> is <diff> minus the memory usage of nested code fragment
 *         guarded with mem_spy_begin and mem_spy_end; 
 *
 * You can use mem_spy_begin and mem_spy_end in functions even if mem_spy_init 
 * and mem_spy_reset are not used on all paths leading to them. 
 * On these paths, no output will be produced (this is what the variable 
 * MemSpyUsed is used for).
 *
 * Notice that MEM_SPY does not rely on any other package, except stdio, 
 * string and malloc.  Thus, you can use it wherever you want.
 *
 * TailleDeMaPile = Max number of nested (mem_spy_begin, mem_spy_end).
 * Please change this number accordingly to your needs. 
 *
 */

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <assert.h>

extern int etext;

/* My own stack (do not rely on any package!) */
#define TailleDeMaPile 20
typedef struct _MemSpyStack 
{
    int index;
    double elt[TailleDeMaPile];
} _MemSpyStack, *MemSpyStack;


/* To store the memory size at program section entry */
static MemSpyStack current_size_stack = NULL;

/* To store the cumulated memory size of inner guarded sections */
static MemSpyStack cumul_size_stack = NULL;

/* To print only wanted results 
 * (and be sure that stacks have been initialized :-) 
*/
static int MemSpyUsed = 0;


void mem_spy_init()
{
    MemSpyUsed = 1;
    current_size_stack = (MemSpyStack) malloc(sizeof(*current_size_stack));
    current_size_stack->index = -1;
    cumul_size_stack = (MemSpyStack) malloc(sizeof(*cumul_size_stack));    
    cumul_size_stack->index = -1;
}

void mem_spy_reset()
{
    if (MemSpyUsed == 1)
    {
	MemSpyUsed = 0;
	if (current_size_stack != NULL)
	    free(current_size_stack);
	if (cumul_size_stack != NULL)
	    free(cumul_size_stack);	    
    }
}

void mem_spy_begin()
{
    
    /* Do nothing if not between <mem_spy_init> ... <mem_spy_reset> */
    if (MemSpyUsed == 1)
    {
	/* Do not go beyond stack limits */
	assert(current_size_stack->index < TailleDeMaPile-1);

	/* Push the current memory size */
	current_size_stack->index += 1;
	current_size_stack->elt[current_size_stack->index] = 
	    (double) ((sbrk(0) - etext)/(double)(1 << 20));

	/* Push 0 on the cumul_size_stack, since there are currently
         * no nested guarded code fragment 
	 */
	cumul_size_stack->index += 1;
	cumul_size_stack->elt[cumul_size_stack->index] = (double) 0;
    }
}

void mem_spy_end(s)
char * s;
{
    double current_begin, current_end, cumul, diff, proper;
    int i;

    /* Do nothing if not between <mem_spy_init> ... <mem_spy_reset> */
    if (MemSpyUsed == 1)
    {
	/* Pop memory size at entry of the code fragment */
	current_end = (sbrk(0) - etext)/(double)(1 << 20);
	current_begin = current_size_stack->elt[current_size_stack->index];
	current_size_stack->index -= 1;
	
	/* Pop cumulated memory size of nested guarded code fragments */
	cumul = cumul_size_stack->elt[cumul_size_stack->index];
	cumul_size_stack->index -= 1;
	
	/* Calculate consumed and proper memory sizes */
	diff = current_end - current_begin;
	proper = diff - cumul; 
	
	/* Prettyprint the results */
	for (i=0; i<=current_size_stack->index; i++)
	    fprintf(stderr, "  ");

	fprintf(stderr, "MEM_SPY "
		"[%s] begin: %10.3fMB end: %10.3fMB diff: %10.3fMB "
		"proper: %10.3fMB\n",
		s, current_begin, current_end, diff, proper);
    
	/* Add the consumed memory size to the previous level cumulated
         * memory size of nested guarded code fragments.
	 */
	if (cumul_size_stack->index != -1)
	    cumul_size_stack->elt[cumul_size_stack->index] += diff;
    }
}



