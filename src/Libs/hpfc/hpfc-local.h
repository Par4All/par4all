/*
 * local definitions
 */

#include <ctype.h>

#include "genC.h"
#include "mapping.h"

#include "ri.h"
#include "text-util.h"
#include "hpf.h"
#include "hpf_private.h"
#include "message.h"

/* added because of mapping.h 
#include "database.h"
#include "resources.h"
#include "pipsdbm.h"
 */

/*
 * prefixes to be used for the variables used in the Psystems
 *
 * Feb 21 1994
 */

#define THETA_PREFIX "THETA"
#define PSI_PREFIX "PSI"
#define GAMMA_PREFIX "GAMMA"
#define DELTA_PREFIX "DELTA"
#define LPHI_PREFIX "LPHI"

#define HPFC_PACKAGE "HPFC-PACKAGE"

/*
 * debug macro
 */
#define IFDBPRINT(n, func, module, stat)                            \
    ifdebug(n)                                                      \
    {                                                               \
       fprintf(stderr,                                              \
	       "[%s] %s statement:\n",                              \
	       func,entity_name(module));                           \
       print_text(stderr,text_statement(module,0,stat));            \
    }

/*
 * Global variables
 */

/* in compiler.c */
extern int 
    uniqueintegernumber,
    uniquefloatnumber,
    uniquelogicalnumber,
    uniquecomplexnumber;

extern entity_mapping 
    oldtonewhostvar,
    oldtonewnodevar,
    newtooldhostvar,
    newtooldnodevar,
    hpfnumber, 
    hpfalign, 
    hpfdistribute;

extern statement_mapping
    hostgotos,
    nodegotos;

extern list 
    distributedarrays,
    templates,
    processors;	

extern entity
    hostmodule,
    nodemodule;

/* in compile-decl.c */

extern entity_mapping
    newdeclarations;

#define computer reference
extern computer 
    currentcomputer; 
/*extern list currentloopindexes; */

/* in run-time.c */

extern entity 
    e_MYPOS, /* used in the overlap analysis, defined in compile.c */
    e_LoopBounds;

/* in o-analysis.c */

extern list
    lloop;

