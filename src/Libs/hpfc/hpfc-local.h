/*
 * local definitions
 * 
 * SCCS stuff:
 * $RCSfile: hpfc-local.h,v $ ($Date: 1995/02/27 18:35:16 $, )
 * version $Revision$,
 * got on %D%, %T%
 * $Id$
 */

#include <ctype.h>

#include "genC.h"
#include "mapping.h"

#include "ri.h"
#include "text-util.h"
#include "hpf.h"
#include "hpf_private.h"
#include "message.h"

#define HPFC_PACKAGE "HPFC-PACKAGE"

/*
 * Global variables
 */

/* in compiler.c */

extern entity
    host_module,
    node_module;
