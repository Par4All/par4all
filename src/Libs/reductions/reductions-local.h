/* $RCSfile: reductions-local.h,v $ (version $Revision$)
 * $Date: 1996/06/21 17:50:09 $, 
 */
/* shorthands for REDUCTION:
 */
#define reduction_variable(r) reference_variable(reduction_reference(r))
#define reduction_none_p(r) reduction_operator_none_p(reduction_op(r))

/* quick debug macros
 */
#define DEBUG_REDUCTION(level, msg, red) \
  ifdebug(level){pips_debug(level, msg); print_reduction(red);}
#define DEBUG_REDUCTIONS(level, msg, reds) \
  ifdebug(level){pips_debug(level, msg); \
                 gen_map(print_reduction, reductions_list(reds));}

/* end of $RCSfile: reductions-local.h,v $
 */
