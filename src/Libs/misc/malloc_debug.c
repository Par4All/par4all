/* #include <sys/stdtypes.h>*/
/* #include "malloc.h" */

#include "genC.h"

#include "misc.h"

void pips_malloc_debug()
{
    debug_on("MALLOC_DEBUG_LEVEL");
#if __GNUC__foo || sunfoo
    if (get_debug_level()==9) {
	debug(9, "pips_malloc_debug", 
	      "malloc_debug level of error diagnosis is 2\n");
	malloc_debug(2);
    }
    else if (get_debug_level()>=5) {
	debug(5, "pips_malloc_debug", "malloc(50) returns %x\n", 
	      (int) malloc(50));
	debug(5, "pips_malloc_debug", "call to malloc_verify()\n");
	pips_assert("pips_malloc_debug", malloc_verify());
 	malloc_debug(1);
    }
    else if (get_debug_level()>=1) 
	malloc_debug(1);
    else
	malloc_debug(0);
#else
    debug(1, "pips_malloc_debug", "No malloc_debug on this system\n");
#endif
    debug_off();
}
