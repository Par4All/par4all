#define EXTERNALIZED_CODE_PRAGMA_BEGIN "BEGIN_KAAPI_%s"
#define EXTERNALIZED_CODE_PRAGMA_END "END_KAAPI_%s"
#define EXTERNALIZED_CODE_PRAGMA_ANALYZED_TOP "ANALYZED_KAAPI_%s (%d statements)"
#define EXTERNALIZED_CODE_PRAGMA_ANALYZED_BOTTOM "ANALYZED_KAAPI_%s (%d statements)"
#define EXTERNALIZED_CODE_PRAGMA_ANALYZED_PREFIX_TOP "--> "
#define EXTERNALIZED_CODE_PRAGMA_ANALYZED_PREFIX_BOTTOM "<-- "

#define EXTERNALIZED_FUNCTION_PARAM_NAME "%s_PARAM_%d"
#define EXTERNALIZED_FUNCTION_PRIVATE_PARAM_NAME "%s_PRIV"

#define EXTERNALIZED_CODE_PRAGMA_CALL "CALL_KAAPI_%s"

#include <string.h>
#include <stdio.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "properties.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "effects-simple.h"
#include "transformer.h"
#include "transformations.h"
#include "control.h"
#include "misc.h"
#include "prettyprint.h"


typedef struct 
{
  string searched_string;
  list list_of_statements;
} statement_checking_context;

typedef struct
{
  statement searched_statement;
  statement found_sequence_statement;
} sequence_searching_context;

typedef struct 
{
  entity searched_entity;
  entity replacement_entity;
} replace_reference_context;
