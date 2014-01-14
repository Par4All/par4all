/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/* MEM_SPY : Package to track memory usage.
 * Beatrice Creusillet - August 1995 
 */
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
 * Modifications:
 *  - verbosity parameter added
 *  - sbrk(0) replaced by mallinfo() to obtain more precise information
 *  - new function mem_spy_info() to print mallinfo
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

/* this is **not** portable */
/* use wait3 sys call or whatever */
/* extern int etext; */

#include "misc-local.h"

#define ABS(x) ((x>0)?(x):-(x))

/* My own stack (do not rely on any package!) */
#define TailleDeMaPile 30
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

/* To control verbosity: from 2, max, to -1, shut up */
static int verbosity = 2;

/* To control granularity: threshold in MB for reporting changes in memory 
 * use. 0.004 is about one page.
 */
static double granularity = 0.;

/* measurement chosen */
static measurement_type measurement = NET_MEASURE;

/* log2 of unit chosen: 0 = Byte, 10 = KB, 20 = MB */
static int log_of_unit = 0;

void mem_spy_init(int v, double g, measurement_type t, int unit)
{
    assert(-1<=v && v<=2);
    assert(0<=unit && unit<=20);
    verbosity = v;
    granularity = g;
    measurement = t;
    log_of_unit = unit;

    MemSpyUsed = 1;
    current_size_stack = (MemSpyStack) malloc(sizeof(*current_size_stack));
    current_size_stack->index = -1;
    cumul_size_stack = (MemSpyStack) malloc(sizeof(*cumul_size_stack));    
    cumul_size_stack->index = -1;
}

void mem_spy_reset()
{
    /* Too many dynamic calls to mem_spy_begin()? */
    if(current_size_stack->index!=-1) {
	fprintf(stderr,
		"Too many calls to mem_spy_begin(), non-empty stack reset\n");
    }

    if (MemSpyUsed == 1)
    {
	MemSpyUsed = 0;
	if (current_size_stack != NULL)
	    free(current_size_stack);
	if (cumul_size_stack != NULL)
	    free(cumul_size_stack);
    }
    verbosity = 2;
}

static int 
current_memory_size()
{
    /* struct mallinfo heap_info; */
    int memory_size;
    
    /* heap_info = mallinfo(); */

    switch(measurement) {
    case SBRK_MEASURE: 
	memory_size = /* sbrk(0) - etext*/ -1;
	break;
    case NET_MEASURE: 
	/* memory_size = heap_info.uordblks-8*heap_info.ordblks; */
	memory_size = -1;
	break;
    case GROSS_MEASURE: 
	/* memory_size = heap_info.uordbytes; */
	memory_size = -1;
	break;
    default:
	abort();
    }

    return memory_size;
}

void mem_spy_begin()
{
    
    /* Do nothing if not between <mem_spy_init> ... <mem_spy_reset> */
    if (MemSpyUsed == 1)
    {
	int memory_size = current_memory_size();

	/* Do not go beyond stack limits */
	assert(current_size_stack->index < TailleDeMaPile-1);

	/* Push the current memory size */
	current_size_stack->index += 1;

	current_size_stack->elt[current_size_stack->index] = 
	    (double) (memory_size/(double)(1 << log_of_unit));

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
	int memory_size = current_memory_size();
	char * format;

	/* Too many dynamic calls to mem_spy_end()? */
	if(current_size_stack->index<0) {
	    fprintf(stderr,
		    "Too many calls to mem_spy_end(), stack underflow\n");
	    abort();
	}

	switch(log_of_unit) {
	case 0: format =
	    "MEM_SPY "
		"[%s] begin: %10.0f B end: %10.0f B diff: %10.0f B "
		    "proper: %10.0f B\n";
	    break;
	case 10: format =
	    "MEM_SPY "
		"[%s] begin: %10.3f KB end: %10.3f KB diff: %10.3f KB "
		    "proper: %10.3f KB\n";
	    break;
	case 20: format =
	    "MEM_SPY "
		"[%s] begin: %10.6f MB end: %10.6f MB diff: %10.6f MB "
		    "proper: %10.6f MB\n";
	    break;
	default: format =
	    "MEM_SPY (unknown unit %d) "
		"[%s] begin: %10.6f (?) end: %10.6f (?) diff: %10.6f (?) "
		    "proper: %10.6f (?)\n";
	    break;
	}

	/* Pop memory size at entry of the code fragment */
	current_end = memory_size/(double)(1 << log_of_unit);
	current_begin = current_size_stack->elt[current_size_stack->index];
	current_size_stack->index -= 1;
	
	/* Pop cumulated memory size of nested guarded code fragments */
	cumul = cumul_size_stack->elt[cumul_size_stack->index];
	cumul_size_stack->index -= 1;
	
	/* Calculate consumed and proper memory sizes */
	diff = current_end - current_begin;
	proper = diff - cumul; 
	
	/* if verbosity==-1, nothing is printed */
	if(verbosity==2 ||
	   (verbosity==1 && ABS(diff) >= granularity) ||
	   (verbosity==0 && ABS(proper) >= granularity)) {
	    /* Prettyprint the results */
	    for (i=0; i<=current_size_stack->index; i++)
		fprintf(stderr, "  ");

	    fprintf(stderr, format,
		    s, current_begin, current_end, diff, proper);
	}

	/* Add the consumed memory size to the previous level cumulated
         * memory size of nested guarded code fragments.
	 */
	if (cumul_size_stack->index != -1)
	    cumul_size_stack->elt[cumul_size_stack->index] += diff;
    }
}

/* To print mallinfo, for debugging memory leaks*/
void mem_spy_info()
{
    /* 
       struct mallinfo heap_info = mallinfo();

    fprintf(stderr, "total space in arena: \t%d", heap_info.arena);
    fprintf(stderr, "number of ordinary blocks: \t%d", heap_info.ordblks);
    fprintf(stderr, "number of small blocks: \t%d", heap_info.smblks);
    fprintf(stderr, "number of holding blocks: \t%d", heap_info.hblks);
    fprintf(stderr, "space in holding block headers: \t%d", heap_info.hblkhd);
    fprintf(stderr, "space in small blocks in use: \t%d", heap_info.usmblks);
    fprintf(stderr, "space in free small blocks: \t%d", heap_info.fsmblks);
    fprintf(stderr, "space in ordinary blocks in use: \t%d", heap_info.uordblks);
    fprintf(stderr, "space in free ordinary blocks: \t%d", heap_info.fordblks);
    fprintf(stderr, "cost of enabling keep option: \t%d", heap_info.keepcost);
    fprintf(stderr, "max size of small blocks: \t%d", heap_info.mxfast);
    fprintf(stderr, "number of small blocks in a holding block: \t%d", heap_info.nlblks);
    fprintf(stderr, "small block rounding factor: \t%d", heap_info.grain);
    fprintf(stderr, "space (including overhead) allocated in ord. blks: \t%d", heap_info.uordbytes);
    fprintf(stderr, "number of ordinary blocks allocated: \t%d", heap_info.allocated);
    fprintf(stderr, "bytes used in maintaining the free tree: \t%d", heap_info.treeoverhead);
    */
}
