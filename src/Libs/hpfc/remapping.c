/* HPFC module by Fabien COELHO
 *
 * $RCSfile: remapping.c,v $ version $Revision$
 * ($Date: 1995/04/19 11:13:04 $, ) 
 */

#include "defines-local.h"

void remapping_compile(s, hsp, nsp)
statement s, *hsp /* Host Statement Pointer */, *nsp /* idem Node */;
{
    user_warning("remapping compile", "not implemented yet\n");

    *hsp = make_continue_statement(entity_empty_label());
    *nsp = make_continue_statement(entity_empty_label());
}

/* that is all
 */
