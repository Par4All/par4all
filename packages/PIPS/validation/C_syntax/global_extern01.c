/* The externe declaration is invalidated by the direct declaration */

extern int a;

/* a is not static, but it is not fully external as it is declared and
   allocated and potentially initialized in this module. */
int a;
