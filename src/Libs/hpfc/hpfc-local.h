/* local definitions
 * 
 * $RCSfile: hpfc-local.h,v $ ($Date: 1995/03/27 16:22:35 $, )
 * version $Revision$,
 */

/* hmmm... shouldn't be necessary
 */
#include <ctype.h>
#include "genC.h"
#include "ri.h"
#include "text-util.h"
#include "hpf.h"
#include "hpf_private.h"
#include "message.h"

#define HPFC_PACKAGE "HPFC-PACKAGE"

#define hpfc_warning \
    if (!get_bool_property("HPFC_NO_WARNING")) user_warning

extern entity /* in compiler.c */
    host_module,
    node_module;

/*  end of $RCSfile: hpfc-local.h,v $
 */
