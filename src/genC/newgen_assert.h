/* abort version of assert.
 * the message generates the function name if possible.
 * message_assert prints a message before aborting
 *
 * $RCSfile: newgen_assert.h,v $ ($Date: 1995/09/22 13:09:23 $, )
 * version $Revision$
 * got on %D%, %T%
 */

#ifdef __GNUC__
#define _newgen_assert_message \
    "[%s] (%s:%d) assertion failed\n", __FUNCTION__, __FILE__, __LINE__
#else
#define _newgen_assert_message \
    "Assertion failed (%s:%d)\n", __FILE__, __LINE__
#endif

#ifndef NDEBUG
#define assert(ex) \
  {\
      if (!(ex)) \
      {\
         (void) fprintf(stderr, _newgen_assert_message);\
	 (void) abort();\
      }\
   }
#define message_assert(msg, ex) \
  {\
      if (!(ex)) \
      {\
         (void) fprintf(stderr, _newgen_assert_message);\
         (void) fprintf(stderr, "\n %s not verified\n\n", msg);\
	 (void) abort();\
      }\
   }
#else
#define assert(ex)
#define message_assert(msg, ex)
#endif

/*  That is all
 */
