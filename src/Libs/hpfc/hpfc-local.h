/*
 * local definitions
 * 
 * SCCS stuff:
 * $RCSfile: hpfc-local.h,v $ ($Date: 1994/06/03 14:14:45 $, )
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

/*
 * prefixes to be used for the variables used in the Psystems
 *
 * Feb 21 1994
 */

#define ALPHA_PREFIX "ALPHA"
#define LALPHA_PREFIX "LALPHA"
#define THETA_PREFIX "THETA"
#define PSI_PREFIX "PSI"
#define GAMMA_PREFIX "GAMMA"
#define DELTA_PREFIX "DELTA"
#define IOTA_PREFIX "IOTA"
#define SIGMA_PREFIX "SIGMA"

#define HPFC_PACKAGE "HPFC-PACKAGE"

/*
 * Global variables
 */

/* in compiler.c */

extern statement_mapping
    hostgotos,
    nodegotos;

extern entity
    host_module,
    node_module;

/* in compile-decl.c */

#define computer reference

extern list
    lloop;

