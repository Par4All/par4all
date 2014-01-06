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
#include "stdio.h"
#include "stdlib.h"

/* mallinfo portability os restricted to SUN C library */
void
print_malloc_info(__attribute__((unused)) FILE * fd)
{
/*
    static struct mallinfo m;

    m = mallinfo();

    user_log("print_malloc_info", "T: %d      U: %d      F: %d\n", 
	    m.arena, m.uordblks, m.fordblks);
	    */
}

void print_full_malloc_info(__attribute__((unused)) FILE * fd)
{
/*
    static struct mallinfo m;

    m = mallinfo();

    user_log("print_full_malloc_info", 
	     "total space in arena:                              %d\n", 
	    m.arena);
    user_log("print_full_malloc_info", 
	     "number of ordinary blocks:                         %d\n", 
	    m.ordblks); 
    user_log("print_full_malloc_info", 
	     "number of small blocks:                            %d\n", 
	    m.smblks);
    user_log("print_full_malloc_info", 
	     "number of holding blocks:                          %d\n", 
	    m.hblks);
    user_log("print_full_malloc_info", 
	     "space in holding block headers:                    %d\n", 
	    m.hblkhd);
    user_log("print_full_malloc_info", 
	     "space in small blocks in use:                      %d\n", 
	    m.usmblks);
    user_log("print_full_malloc_info", 
	     "space in free small blocks:                        %d\n", 
	    m.fsmblks);
    user_log("print_full_malloc_info", 
	     "space in ordinary blocks in use:                   %d\n", 
	    m.uordblks);
    user_log("print_full_malloc_info", 
	     "space in free ordinary blocks:                     %d\n", 
	    m.fordblks);
    user_log("print_full_malloc_info", 
	     "cost of enabling keep option:                      %d\n", 
	    m.keepcost);
#ifdef sun
    user_log("print_full_malloc_info", 
	     "max size of small blocks:                          %d\n", 
	    m.mxfast);
    user_log("print_full_malloc_info", 
	     "number of small blocks in a holding block:         %d\n", 
	    m.nlblks);
    user_log("print_full_malloc_info", 
	     "small block rounding factor:                       %d\n", 
	    m.grain);
    user_log("print_full_malloc_info", 
	     "space (including overhead) allocated in ord. blks: %d\n",
	    m.uordbytes);
    user_log("print_full_malloc_info", 
	     "number of ordinary blocks allocated:               %d\n", 
	    m.allocated);
    user_log("print_full_malloc_info", 
	     "bytes used in maintaining the free tree:           %d\n", 
	    m.treeoverhead);
#endif
*/
}
