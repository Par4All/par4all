#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"

bool prepend_comment(string mn) {

  statement s = PIPS_PHASE_PRELUDE(
    mn, "PREPEND_COMMENT_DEBUG_LEVEL");

  string c = get_string_property("PREPEND_COMMENT");

  insert_comments_to_statement(s, c);

  PIPS_PHASE_POSTLUDE(s);
}
