/*
 * $Id$
 *
 * FPIPS stands for Full PIPS, or Fabien PIPS;-)
 *
 * it provides a single executable for {,t,w}pips, enabling faster
 * link when developping and testing. Also a single executable can
 * be exported, reducing the size of binary distributions.
 * The execution depends on the name of the executable, or the first option.
 *
 * C macros of interest: FPIPS_WITHOUT_{,T,W}PIPS
 *
 * FC, Mon Aug 18 09:09:32 GMT 1997
 */


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "genC.h"
#include "misc.h"

/******************************************************************** MACROS */

#if defined(FPIPS_WITHOUT_PIPS)
#define PIPS(c, v) fpips_error("pips", c, v)
#else
extern int pips_main(int, char**);
#define PIPS(c, v) pips_main(c, v)
#endif

#if defined(FPIPS_WITHOUT_TPIPS)
#define TPIPS(c, v) fpips_error("tpips", c, v)
#else
extern int tpips_main(int, char**);
#define TPIPS(c, v) tpips_main(c, v)
#endif

#if defined(FPIPS_WITHOUT_WPIPS)
#define WPIPS(c, v) fpips_error("wpips", c, v)
#else
extern int wpips_main(int, char**);
#define WPIPS(c, v) wpips_main(c, v)
#endif


/******************************************************************** UTILS */

static int fpips_usage(int ret)
{
    fprintf(stderr, 
	    "Usage: fpips [-hvPTW] (other options and arguments...)\n"
	    "\t-h: this help...\n"
	    "\r-v: version\n"
	    "\t-P: pips\n"
	    "\t-T: tpips\n"
	    "\t-W: wpips\n"
	    "\tdefault: tpips\n");

    return ret;
}

/* non static to avoid a gcc warning if not called.
 */
int fpips_error(char * what, int argc, char ** argv)
{
    fprintf(stderr, "[fpips] sorry, %s not available (" SOFT_ARCH ")\n", what);
    return fpips_usage(1);
}

/* returns wether name ends with ref 
 */
static int name_end_p(char * name, char * ref)
{
    int nlen = strlen(name), rlen = strlen(ref);
    if (nlen<rlen) return FALSE;
    while (rlen>0) 
	if (ref[--rlen]!=name[--nlen]) 
	    return FALSE;
    return TRUE;
}

static int fpips_version(int ret)
{
    fprintf(stderr, "fpips (ARCH=" SOFT_ARCH ", DATE=" UTC_DATE")\n");
    return ret;
}

/********************************************************************* MAIN */

int fpips_main(int argc, char **  argv)
{
    debug_on("FPIPS_DEBUG_LEVEL");
    pips_debug(1, "considering %s for execution\n", argv[0]);
    debug_off();

    if (argc<1) return TPIPS(argc, argv); /* should not happen */

    /* According to the shell or the debugger, the path may be
       complete or not... RK. */
    if (name_end_p(argv[0], "tpips")) return TPIPS(argc, argv);
    if (name_end_p(argv[0], "wpips")) return WPIPS(argc, argv);
    if (name_end_p(argv[0], "/pips") || strcmp(argv[0], "pips") == 0)
	return  PIPS(argc, argv);

    /* else look for the first option, if any.
     */
    if (argc<2) TPIPS(argc, argv);

    /* options
     */
    if (same_string_p(argv[1], "-h")) 
	return fpips_usage(0);

    if (same_string_p(argv[1], "-v"))
	return fpips_version(0);

    if (same_string_p(argv[1], "-P")) {
	/* Do not forget to update what will become the new
           argv[0]. Especially for X11 or XView in WPips which parse
           the arguments later... RK. */
	argv[1] = argv[0];
	return  PIPS(argc-1, argv+1);
    }
    if (same_string_p(argv[1], "-T")) {
	argv[1] = argv[0];
	return TPIPS(argc-1, argv+1);
    }
    if (same_string_p(argv[1], "-W")) {
	argv[1] = argv[0];
	return WPIPS(argc-1, argv+1);
    }
    /* else try tpips...
     */
    return TPIPS(argc, argv);
}
