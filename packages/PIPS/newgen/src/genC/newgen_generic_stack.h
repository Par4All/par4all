/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of NewGen.

  NewGen is free software: you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or any later version.

  NewGen is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
  License for more details.

  You should have received a copy of the GNU General Public License along with
  NewGen.  If not, see <http://www.gnu.org/licenses/>.

*/
/*
   Made after the generic current mappings
   Fabien COELHO, 05/12/94
*/

#define DEFINE_STACK(PREFIX, name, type)				\
  /* the stack */							\
  static stack name##_stack = stack_undefined;				\
  /* its functions */							\
  PREFIX void __attribute__ ((unused)) make_##name##_stack(void)	\
  {									\
    assert(name##_stack==stack_undefined);				\
    name##_stack = stack_make(type##_domain, 0, 0);			\
  }									\
  PREFIX void __attribute__ ((unused)) free_##name##_stack(void)	\
  {									\
    stack_free(&name##_stack);						\
    name##_stack = stack_undefined;					\
  }									\
  PREFIX stack __attribute__ ((unused)) get_##name##_stack(void)	\
  {									\
    return name##_stack;						\
  }									\
  PREFIX void __attribute__ ((unused)) set_##name##_stack(stack s)	\
  {									\
    assert(name##_stack==stack_undefined);				\
    name##_stack = s;							\
  }									\
  PREFIX void __attribute__ ((unused)) reset_##name##_stack(void)	\
  {									\
    assert(name##_stack!=stack_undefined);				\
    name##_stack = stack_undefined;					\
  }									\
  PREFIX void __attribute__ ((unused)) name##_push(type i)		\
  {									\
    stack_push((void *)i, name##_stack);				\
  }									\
  PREFIX bool __attribute__ ((unused)) name##_filter(type i)		\
  {									\
    stack_push((void *)i, name##_stack);				\
    return true;							\
  }									\
  PREFIX void __attribute__ ((unused)) name##_rewrite(type i)		\
  {									\
    type __attribute__ ((unused)) t = (type)stack_pop(name##_stack);\
    assert(t==i);				\
  }									\
  PREFIX type __attribute__ ((unused)) name##_replace(type i)		\
  {									\
    return (type) stack_replace((void *)i, name##_stack);		\
  }									\
  PREFIX type __attribute__ ((unused)) name##_pop(void)			\
  {									\
    return (type) stack_pop(name##_stack);				\
  }									\
  PREFIX type __attribute__ ((unused)) name##_head(void)		\
  {									\
    return (type) stack_head(name##_stack);				\
  }									\
  PREFIX type __attribute__ ((unused)) name##_nth(int n)		\
  {									\
    return (type)  stack_nth(name##_stack, n);				\
  }									\
  PREFIX bool __attribute__ ((unused)) name##_empty_p(void)		\
  {									\
    return stack_empty_p(name##_stack);					\
  }									\
  PREFIX int __attribute__ ((unused)) name##_size(void)			\
  {									\
    return stack_size(name##_stack);					\
  }									\
  PREFIX void __attribute__ ((unused)) error_reset_##name##_stack(void)	\
  {									\
    name##_stack = stack_undefined;					\
  }

#define DEFINE_LOCAL_STACK(name, type) DEFINE_STACK(static, name, type)
#define DEFINE_GLOBAL_STACK(name, type) DEFINE_STACK(extern, name, type)

/*  That is all
 */
