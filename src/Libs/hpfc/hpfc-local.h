/*
 * local definitions
 * 
 * SCCS stuff:
 * $RCSfile: hpfc-local.h,v $ ($Date: 1995/03/22 11:13:09 $, )
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

extern entity /* in compiler.c */
    host_module,
    node_module;

extern void hpfc_warning();
