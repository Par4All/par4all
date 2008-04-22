/* 
 * $Id$
 *
 * managing arithmetic errors...
 * detecting and managing arithmetic errors on Values should be
 * systematic. These macros gives a C++ look and feel to this
 * management. 
 *
 * (c) CA et FC, Sept 1997
 */

#if !defined(linear_arithmetic_error_included)
#define linear_arithmetic_error_included

#include <setjmp.h>

typedef void (*exception_callback_t)(char const *, char const *, int const);

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

/* 'const' out because of cproto 4.6. FC 13/06/2003 */
#define EXCEPTION extern linear_exception_t

#define THROW(what) \
   (throw_exception(what, __CURRENT_FUNCTION_NAME__, __FILE__, __LINE__))

#define CATCH(what) 							\
   if (setjmp(*push_exception_on_stack(what, __CURRENT_FUNCTION_NAME__,	\
				     __FILE__, __LINE__)))

#define UNCATCH(what)						\
     (pop_exception_from_stack(what, __CURRENT_FUNCTION_NAME__,	\
			       __FILE__, __LINE__))

#define TRY else

#define RETHROW() THROW(the_last_just_thrown_exception)

#endif /* linear_arithmetic_error_included */

/* end of it.
 */
