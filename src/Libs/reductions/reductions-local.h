/* $RCSfile: reductions-local.h,v $ (version $Revision$)
 * $Date: 1996/06/15 13:22:10 $, 
 */
/* shorthands for REDUCTION:
 */
#define reduction_variable(r) reference_variable(reduction_reference(r))

/* shorthands for EFFECT:
 */
#define effect_variable(e) reference_variable(effect_reference(e))
#define effect_write_p(e) action_write_p(effect_action(e))
#define effect_read_p(e) action_read_p(effect_action(e))

/* end of $RCSfile: reductions-local.h,v $
 */
