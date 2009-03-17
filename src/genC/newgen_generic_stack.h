/* Made after the generic current mappings
 * Fabien COELHO, 05/12/94
 *
 * $Id$
 */

#define DEFINE_STACK(PREFIX, name, type)			\
  /* the stack */						\
  static stack name##_stack = stack_undefined;			\
  /* its functions */						\
  PREFIX void make_##name##_stack(void)				\
  {								\
    assert(name##_stack==stack_undefined);			\
    name##_stack = stack_make(type##_domain, 0, 0);		\
  }								\
  PREFIX void free_##name##_stack(void)				\
  {								\
    stack_free(&name##_stack);					\
    name##_stack = stack_undefined;				\
  }								\
  PREFIX stack get_##name##_stack(void)				\
  {								\
    return name##_stack;					\
  }								\
  PREFIX void set_##name##_stack(stack s)			\
  {								\
    assert(name##_stack==stack_undefined);			\
    name##_stack = s;						\
  }								\
  PREFIX void reset_##name##_stack(void)			\
  {								\
    assert(name##_stack!=stack_undefined);			\
    name##_stack = stack_undefined;				\
  }								\
  PREFIX void name##_push(type i)				\
  {								\
    stack_push((void *)i, name##_stack);			\
  }								\
  PREFIX bool name##_filter(type i)				\
  {								\
    stack_push((void *)i, name##_stack);			\
    return TRUE;						\
  }								\
  PREFIX void name##_rewrite(type i)				\
  {								\
    assert((type)stack_pop(name##_stack)==i);			\
  }								\
  PREFIX type name##_replace(type i)				\
  {								\
    return (type) stack_replace((void *)i, name##_stack);	\
  }								\
  PREFIX type name##_pop(void)					\
  {								\
    return (type) stack_pop(name##_stack);			\
  }								\
  PREFIX type name##_head(void)					\
  {								\
    return (type) stack_head(name##_stack);			\
  }								\
  PREFIX bool name##_empty_p(void)				\
  {								\
    return stack_empty_p(name##_stack);				\
  }								\
  PREFIX int name##_size(void)					\
  {								\
    return stack_size(name##_stack);				\
  }								\
  PREFIX void error_reset_##name##_stack(void)			\
  {								\
    name##_stack = stack_undefined;				\
  }								\
  static void check_##name##_stack(void)			\
  {								\
    stack s = get_##name##_stack();				\
    void							\
      *item_1 = (void *) check_##name##_stack,			\
      *item_2 = (void *) get_##name##_stack;			\
    reset_##name##_stack();					\
    make_##name##_stack();					\
    assert(name##_empty_p());					\
    name##_push((type) item_1);					\
    assert((void *) name##_head()==item_1);			\
    name##_replace((type) item_2);				\
    assert((void *) name##_pop()==item_2);			\
    assert(name##_filter((type) item_1));			\
    name##_rewrite((type) item_1);				\
    assert(name##_size()==0);					\
    free_##name##_stack();					\
    reset_##name##_stack();					\
    error_reset_##name##_stack();				\
    set_##name##_stack(s);					\
  }

#define DEFINE_LOCAL_STACK(name, type) DEFINE_STACK(static, name, type)
#define DEFINE_GLOBAL_STACK(name, type) DEFINE_STACK(extern, name, type)

/*  That is all
 */
