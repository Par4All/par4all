/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/*
 * FPIPS stands for Full PIPS, or Fabien PIPS;-)
 *
 * it provides a single executable for {,t,w}pips, enabling faster
 * link when developing and testing. Also a single executable can
 * be exported, reducing the size of binary distributions.
 * The execution depends on the name of the executable, or the first option.
 *
 * C macros of interest: FPIPS_WITHOUT_{,G,T,W}PIPS to disable some versions.
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

#if defined(FPIPS_WITHOUT_GPIPS)
#define GPIPS(c, v) fpips_error("gpips", c, v)
#else
extern int gpips_main(int, char**);
#define GPIPS(c, v) gpips_main(c, v)
#endif

#define USAGE							\
    "Usage: fpips [-hvGPTW] (other options and arguments...)\n"	\
    "\t-h: this help...\n"					\
    "\t-v: version\n"						\
    "\t-G: gpips\n"						\
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
  fprintf(stderr, "[fpips] (ARCH=" STRINGIFY(SOFT_ARCH) ", DATE=" STRINGIFY(UTC_DATE) ")\n\n");
    return ret;
}

/* non static to avoid a gcc warning if not called.
 */
int fpips_error(char * what,
		int __attribute__ ((unused)) argc,
		char __attribute__ ((unused)) ** argv)
{
    fprintf(stderr, "[fpips] sorry, %s not available (" STRINGIFY(SOFT_ARCH) ")\n", what);
    return fpips_usage(1);
}

/* returns whether name ends with ref
 */
static int name_end_p(char * name, char * ref)
{
    int nlen = strlen(name), rlen = strlen(ref);
    if (nlen<rlen) return false;
    while (rlen>0)
	if (ref[--rlen]!=name[--nlen])
	    return false;
    return true;
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
    if (name_end_p(argv[0], "gpips"))
	return GPIPS(argc, argv);
    if (name_end_p(argv[0], "tpips"))
	return TPIPS(argc, argv);
    if (name_end_p(argv[0], "wpips"))
	return WPIPS(argc, argv);
    if (name_end_p(argv[0], "/pips") || same_string_p(argv[0], "pips"))
	return  PIPS(argc, argv);

    /* parsing of options may be continuate by called version.
     */
    while ((opt = getopt(argc, argv, "hvGPTW"))!=-1)
    {
	switch (opt)
	{
	case 'h': fpips_usage(0); break;
	case 'v': fpips_version(0); break;
	case 'G': return GPIPS(argc, argv);
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
