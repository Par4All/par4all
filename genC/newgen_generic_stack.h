/* Made after the generic current mappings
 * Fabien COELHO, 05/12/94
 */

#define DEFINE_STATIC_STACK(name, type) \
\
static gen_stack name##_stack = gen_stack_undefined; \
\
void make_##name##_stack() \
{\
  assert(name##_stack==gen_stack_undefined);\
  name##_stack = gen_stack_make(type##_domain, 0);\
}\
\
void free_##name##_stack() \
{\
  gen_stack_free(name##_stack); \
  name##_stack = gen_stack_undefined; \
}\
\
gen_stack get_##name##_stack() \
{\
  return(name##_stack);\
}\
\
void set_##name##_stack(s) \
gen_stack s; \
{\
  assert(name##_stack==gen_stack_undefined);\
  name##_stack = s;\
}\
\
void reset_##name##_stack()\
{\
  name##_stack = gen_stack_undefined; \
}\
\
void name##_push(i)\
type i;\
{\
  gen_push(i, name##_stack);\
}\
\
type name##_pop()\
{\
  return(gen_pop(name##_stack));\
}\
\
type name##_head()\
{\
  return(gen_head(name##_stack));\
}

/*  That is all
 */
