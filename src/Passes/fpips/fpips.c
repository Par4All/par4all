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

#define true  (1)
#define false (0)

#define same_string_p(s1, s2) (strcmp((s1),(s2))==0)


/******************************************************************** UTILS */

static int
fpips_usage(int ret)
{
    fprintf(stderr, 
	    "Usage: fpips [-hPTW] (other options and arguments...)\n"
	    "\t-h: this help...\n"
	    "\t-P: pips\n"
	    "\t-T: tpips\n"
	    "\t-W: wpips\n"
	    "\tdefault: tpips\n");

    return ret;
}

int /* non static to avoid a gcc warning if not called. */
fpips_error(char * what, int argc, char ** argv)
{
    fprintf(stderr, 
	    "[fpips] sorry, %s not available with " SOFT_ARCH "\n", what);

    return fpips_usage(1);
}

/* returns wether name ends with ref 
 */
static int
name_end_p(char * name, char * ref)
{
    int nlen = strlen(name), rlen = strlen(ref);
    if (nlen<rlen) return false;
    while (rlen>0) 
	if (ref[--rlen]!=name[--nlen]) 
	    return false;
    return true;
}


/********************************************************************* MAIN */

int 
fpips_main(int argc, char **  argv)
{
    if (argc<1) return TPIPS(argc, argv); /* should not happen */

    if (name_end_p(argv[0],  "/pips")) return  PIPS(argc, argv);
    if (name_end_p(argv[0], "/tpips")) return TPIPS(argc, argv);
    if (name_end_p(argv[0], "/wpips")) return WPIPS(argc, argv);

    /* else look for the first option, if any.
     */
    if (argc<2) TPIPS(argc, argv);

    /* options
     */
    if (same_string_p(argv[1], "-h")) return  fpips_usage(0);

    if (same_string_p(argv[1], "-P")) return  PIPS(argc-1, argv+1);
    if (same_string_p(argv[1], "-T")) return TPIPS(argc-1, argv+1);
    if (same_string_p(argv[1], "-W")) return WPIPS(argc-1, argv+1);

    /* else try tpips...
     */
    return TPIPS(argc, argv);
}
