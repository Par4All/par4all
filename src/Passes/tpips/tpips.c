#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdarg.h>
#include <setjmp.h>
#include <stdlib.h>

#include "genC.h"
#include "ri.h"
#include "database.h"
#include "graph.h"
#include "makefile.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "constants.h"
#include "resources.h"
#include "pipsmake.h"

#include "top-level.h"

#include "tpips.h"

#define MAX_ARGS 128

#define TPIPS_PROMPT "pips> "


main(argc, argv)
int argc;
char *argv[];
{
    extern jmp_buf pips_top_level;

    debug_on("PIPS_DEBUG_LEVEL");

    initialize_newgen();

    (void) setjmp(pips_top_level);

    tp_parse ();

}
