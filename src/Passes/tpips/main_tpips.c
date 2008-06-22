/* 
 * $Id$
 *
 * This file contains the main for tpips.
 * Please, do not change anything! do any change to tpips_main().
 */

#include "revisions.h"

#define STRINGIFY_SECOND_STAGE(symbol) #symbol
#define STRINGIFY(symbol) STRINGIFY_SECOND_STAGE(symbol)

/* could be shared somewhere? */
char * soft_revisions = 
  "newgen=" STRINGIFY(NEWGEN_REV) 
  ", linear=" STRINGIFY(LINEAR_REV) 
  ", pips=" STRINGIFY(PIPS_REV)
  ", nlpmake=" STRINGIFY(NLPMAKE_REV);

extern char * pips_thanks(char *, char *);
extern int tpips_main(int, char**);

int main(int argc, char ** argv)
{
    pips_thanks("tpips", argv[0]);
    return tpips_main(argc, argv);
}
