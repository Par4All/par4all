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
 * C macros of interest: FPIPS_WITHOUT_{,T,W}PIPS to disable some versions.
 *
 * FC, Mon Aug 18 09:09:32 GMT 1997
 *
 * $Log: fpips.c,v $
 * Revision 1.12  1998/07/25 09:35:52  coelho
 * more comments.
 * verbose when default chosen.
 *
 * Revision 1.11  1998/07/21 17:46:06  coelho
 * options with getopt. -v option added.
 *
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

#define USAGE							\
    "Usage: fpips [-hvPTW] (other options and arguments...)\n"	\
    "\t-h: this help...\n"					\
    "\t-v: version\n"						\
    "\t-P: pips\n"						\
    "\t-T: tpips\n"						\
    "\t-W: wpips\n"						\
    "\tdefault: run as tpips\n\n"

/******************************************************************** UTILS */

/* print out usage informations.
 */
static int fpips_usage(int ret)
{
    fprintf(stderr, USAGE);
    return ret;
}

/* print out fpips version.
 */
static int fpips_version(int ret)
{
    fprintf(stderr, "[fpips] (ARCH=" SOFT_ARCH ", DATE=" UTC_DATE ")\n\n");
    return ret;
}

/* non static to avoid a gcc warning if not called.
 */
int fpips_error(char * what, int argc, char ** argv)
{
    fprintf(stderr, "[fpips] sorry, %s not available (" SOFT_ARCH ")\n", what);
    return fpips_usage(1);
}

/* returns whether name ends with ref 
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

/********************************************************************* MAIN */

int fpips_main(int argc, char **  argv)
{
    int opt;

    debug_on("FPIPS_DEBUG_LEVEL");
    pips_debug(1, "considering %s for execution\n", argv[0]);
    debug_off();

    if (argc<1) return TPIPS(argc, argv); /* should not happen */

    /* According to the shell or the debugger, the path may be
       complete or not... RK. */
    if (name_end_p(argv[0], "tpips")) 
	return TPIPS(argc, argv);
    if (name_end_p(argv[0], "wpips")) 
	return WPIPS(argc, argv);
    if (name_end_p(argv[0], "/pips") || same_string_p(argv[0], "pips"))
	return  PIPS(argc, argv);

    /* parsing of options may be continuate by called version.
     */
    while ((opt = getopt(argc, argv, "hvPTW"))!=-1)
    {
	switch (opt)
	{
	case 'h': fpips_usage(0); break;
	case 'v': fpips_version(0); break;
	case 'P': return PIPS(argc, argv);
	case 'T': return TPIPS(argc, argv);
	case 'W': return WPIPS(argc, argv);
	default:  return fpips_version(1);
	}
    }

    /* else try tpips...
     */
    fprintf(stderr, "[fpips] default: running as tpips\n\n");
    return TPIPS(argc, argv);
}
