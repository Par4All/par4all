/* $Id$
 */
typedef void (*freia_init_func_t)(int, char**);
typedef void (*freia_shutdown_func_t)(void);

extern int freia_register_init_shutdown(char *, freia_init_func_t, freia_shutdown_func_t);
extern void freia_initialize(int, char**);
extern void freia_shutdown(void);
