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
 * Revision 1.27  2000/07/26 09:07:32  coelho
 * *** empty log message ***
 *
 * Revision 1.26  2000/07/26 09:06:32  coelho
 * the_last_just_thrown_exception declared.
 *
 * Revision 1.25  2000/07/26 08:41:40  coelho
 * RETHROW added.
 *
 * Revision 1.24  1998/10/26 14:37:48  coelho
 * constants moved out.
 *
 * Revision 1.23  1998/10/26 14:36:13  coelho
 * constants explicitely defined in .h.
 *
 * Revision 1.22  1998/10/24 15:18:26  coelho
 * THROW macro updated to tell its source.
 *
 * Revision 1.21  1998/10/24 14:33:08  coelho
 * parser exception added.
 *
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

/*
const unsigned int overflow_error = 1;
const unsigned int simplex_arithmetic_error = 2;
const unsigned int user_exception_error = 4;
const unsigned int parser_exception_error = 8;
const unsigned int any_exception_error = ~0;
*/
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

#define THROW(what) \
   (throw_exception(what, __CURRENT_FUNCTION_NAME__, __FILE__, __LINE__))

#define CATCH(what) 							\
   if (setjmp(*push_exception_on_stack(what, __CURRENT_FUNCTION_NAME__,	\
				     __FILE__, __LINE__)))

#define UNCATCH(what)						\
     (pop_exception_from_stack(what, __CURRENT_FUNCTION_NAME__,	\
			       __FILE__, __LINE__))

#define TRY else

unsigned int the_last_just_thrown_exception;
#define RETHROW() THROW(the_last_just_thrown_exception)

#endif /* LINEAR_ARITHMETIC_ERROR_INCLUDED */

/* end of it.
 */
