/* Summary description of the target machine.
 *
 * Should be merged with the wp65 description and the complexity cost model by Lei Zhou
 *
 * Francois Irigoin, 16 January 1993
 */

#include <stdio.h>

#include "genC.h"

int get_cache_line_size(void)
{
    return 1;
}

int get_processor_number(void)
{
    return 16;
}

int get_vector_register_length(void)
{
    return 64;
}

int get_vector_register_number(void)
{
    return 8;
}

int get_minimal_task_size(void)
{
    /* the unit is supposed to be consistent with the complexity cost tables used
     * that should be expressed in machine cycles
     */
    return 10000;
}
