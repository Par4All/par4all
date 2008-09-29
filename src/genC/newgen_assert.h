/* abort version of assert.
 * the message generates the function name if possible.
 * message_assert prints a message before aborting
 *
 * $Id$
 */

#ifdef __GNUC__
#define _newgen_assert_message						\
  "[%s] (%s:%d) assertion failed\n", __FUNCTION__, __FILE__, __LINE__
#else
#define _newgen_assert_message				\
  "Assertion failed (%s:%d)\n", __FILE__, __LINE__
#endif

#ifdef NDEBUG
#define assert(ex)
#define message_assert(msg, ex)
#else
#define assert(ex) {						\
    if (!(ex)) { 						\
      (void) fprintf(stderr, _newgen_assert_message);		\
      (void) abort();						\
    }								\
  }
#define message_assert(msg, ex) {				\
    if (!(ex)) {						\
      (void) fprintf(stderr, _newgen_assert_message);		\
      (void) fprintf(stderr, "\n %s not verified\n\n", msg);	\
      (void) abort();						\
    }								\
  }
#endif /* NDEBUG */

/*  That is all
 */
