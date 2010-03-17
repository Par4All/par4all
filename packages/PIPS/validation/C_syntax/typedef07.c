/* typedef scope in preprocessor: here "cookie" is not a typedef */
/* Case used to avoid quick fixes for typedef06.c */

typedef int f(int cookie);

typedef int (*g)(int);

f typedef07;

/* Not supported by preprocessor */
/* int typedef07(int cookie) {return cookie;} */

int typedef07(int cookie)
{return cookie;}
