/* $RCSfile: reductions-local.h,v $ (version $Revision$)
 * $Date: 1996/06/17 15:52:12 $, 
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

/* end of $RCSfile: reductions-local.h,v $
 */
