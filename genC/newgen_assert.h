/* abort version of assert.
 * the message generates the function name if possible.
 */

#ifdef __GNUC__
#define _newgen_assert_message \
    "[%s] assertion failed, file %s (%d)\n", __FUNCTION__, __FILE__, __LINE__
#else
#define _newgen_assert_message \
    "Assertion failed: file \"%s\", line %d\n", __FILE__, __LINE__
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
#else
#define assert(ex)
#endif
