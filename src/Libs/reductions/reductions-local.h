/* $RCSfile: reductions-local.h,v $ (version $Revision$)
 * $Date: 1996/06/17 18:11:50 $, 
 */
/* shorthands for REDUCTION:
 */
#define reduction_variable(r) reference_variable(reduction_reference(r))
#define reduction_none_p(r) reduction_operator_none_p(reduction_op(r))

/* shorthands for EFFECT:
 */
#define effect_variable(e) reference_variable(effect_reference(e))
#define effect_write_p(e) action_write_p(effect_action(e))
#define effect_read_p(e) action_read_p(effect_action(e))

/* quick debug macros
 */
#define DEBUG_REDUCTION(level, msg, red) \
  ifdebug(level){pips_debug(level, msg); print_reduction(red);}
#define DEBUG_REDUCTIONS(level, msg, reds) \
  ifdebug(level){pips_debug(level, msg); \
                 gen_map(print_reduction, reductions_list(reds));}

/* end of $RCSfile: reductions-local.h,v $
 */
