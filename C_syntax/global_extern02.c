/* The extern declaration is invalidated by the direct declaration */

int a;

/* a is not static, but it is not fully external as it has been declared and
   allocated and potentially initialized in this module. */
extern int a;
