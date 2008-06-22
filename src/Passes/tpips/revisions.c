#include "revisions.h"

#define STRINGIFY_SECOND_STAGE(symbol) #symbol
#define STRINGIFY(symbol) STRINGIFY_SECOND_STAGE(symbol)

/* could be shared somewhere? */
char * soft_revisions = 
  "newgen=" STRINGIFY(NEWGEN_REV) 
  ", linear=" STRINGIFY(LINEAR_REV) 
  ", pips=" STRINGIFY(PIPS_REV)
  ", nlpmake=" STRINGIFY(NLPMAKE_REV);
