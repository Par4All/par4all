/* 
 * $Id$
 *
 * managing arithmetic errors...
 * detecting and managing arithmetic errors on Values should be
 * systematic. These macros gives a C++ look and feel to this
 * management. 
 *
 * (c) CA et FC, Sept 1997
 *
 * $Log: arithmetic_errors.h,v $
 * Revision 1.20  1998/10/24 14:32:45  coelho
 * simpler macros.
 *
 * Revision 1.19  1998/10/24 09:22:47  coelho
 * size update.
 *
 * Revision 1.18  1998/10/24 09:21:45  coelho
 * const added to constants.
 *
 */

#if !defined(LINEAR_ARITHMETIC_ERROR_INCLUDED)
#define LINEAR_ARITHMETIC_ERROR_INCLUDED

#include <setjmp.h>

/* the index points to the first available chunck for a new context...
 */
extern const unsigned int overflow_error;
extern const unsigned int simplex_arithmetic_error;
extern const unsigned int user_exception_error;
extern const unsigned int any_exception_error;

/* use gnu cpp '__FUNCTION__' extension if possible.
 */
#if defined(__GNUC__)
#define __CURRENT_FUNCTION_NAME__ __FUNCTION__
#else
#define __CURRENT_FUNCTION_NAME__ "<unknown>"
#endif

/* set DEBUG_GLOBAL_EXCEPTIONS for debugging information
 */
#define EXCEPTION extern const unsigned int

#define THROW(what) (throw_exception(what))

#define CATCH(what) 							\
   if (setjmp(*push_exception_on_stack(what, __CURRENT_FUNCTION_NAME__,	\
				     __FILE__, __LINE__)))

#define UNCATCH(what)						\
     (pop_exception_from_stack(what, __CURRENT_FUNCTION_NAME__,	\
			       __FILE__, __LINE__))

#define TRY else

#endif /* LINEAR_ARITHMETIC_ERROR_INCLUDED */

/* end of it.
 */
