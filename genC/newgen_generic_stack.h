/* Made after the generic current mappings
 * Fabien COELHO, 05/12/94
 */

#define DEFINE_STACK(PREFIX, name, type) \
\
static gen_stack name##_stack = gen_stack_undefined; \
\
PREFIX void make_##name##_stack() \
{\
  assert(name##_stack==gen_stack_undefined);\
  name##_stack = gen_stack_make(type##_domain, 0);\
}\
\
PREFIX void free_##name##_stack() \
{\
  gen_stack_free(name##_stack); \
  name##_stack = gen_stack_undefined; \
}\
\
PREFIX gen_stack get_##name##_stack() \
{\
  return(name##_stack);\
}\
\
PREFIX void set_##name##_stack(s) \
gen_stack s; \
{\
  assert(name##_stack==gen_stack_undefined);\
  name##_stack = s;\
}\
\
PREFIX void reset_##name##_stack()\
{\
  name##_stack = gen_stack_undefined; \
}\
\
PREFIX void name##_push(i)\
type i;\
{\
  gen_push(i, name##_stack);\
}\
PREFIX type name##_replace(i)\
type i;\
{\
  return(gen_replace(i, name##_stack));\
}\
\
PREFIX type name##_pop()\
{\
  return(gen_pop(name##_stack));\
}\
\
PREFIX type name##_head()\
{\
  return(gen_head(name##_stack));\
}\
\
PREFIX bool name##_empty_p()\
{\
  assert(name##_stack!=gen_stack_undefined);\
  return(gen_stack_empty_p(name##_stack));\
}\
\
PREFIX int name##_size()\
{\
  assert(name##_stack!=gen_stack_undefined);\
  return(gen_stack_size(name##_stack));\
}\
\
static void check_##name()\
{\
  gen_stack s = get_##name##_stack();\
  chunk\
     *item_1 = (chunk *) check_##name,\
     *item_2 = (chunk *) get_##name##_stack;\
  \
  reset_##name##_stack();\
  make_##name##_stack();\
  assert(name##_empty_p());\
  name##_push(item_1);\
  assert(name##_head()==item_1);\
  name##_replace(item_2);\
  assert(name##_pop()==item_2);\
  assert(name##_size()==0);\
  free_##name##_stack();\
  reset_##name##_stack();\
  set_##name##_stack(s);\
}

#define DEFINE_LOCAL_STACK(name, type) DEFINE_STACK(static, name, type)
#define DEFINE_GLOBAL_STACK(name, type) DEFINE_STACK(/**/, name, type)

/*  That is all
 */
