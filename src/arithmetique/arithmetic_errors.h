/* $RCSfile: arithmetic_errors.h,v $ (version $Revision$)
 * $Date: 1996/08/09 16:31:44 $, 
 *
 * managing arithmetic errors...
 * detecting and managing arithmetic errors on Values should be
 * systematic. These macros gives a C++ look and feel to this
 * management. 
 *
 * (c) FC, Aug 1996
 */

#include <setjmp.h>
extern jmp_buf overflow_error;

/* TRY/CATCH/THROW/EXCEPTION: macros with a C++ look and feel.
 * EXCEPTION overflow_error;
 * CATCH(overflow_error) {
 *   ...
 * } TRY {
 *   ... THROW(overflow_error) ...
 * }
 */
#define EXCEPTION jmp_buf
#define CATCH(thrown) if (setjmp(thrown))
#define TRY else
#define THROW(thrown) longjmp(thrown,__LINE__) /* why not!? */

/* end of $RCSfile: arithmetic_errors.h,v $
 */
