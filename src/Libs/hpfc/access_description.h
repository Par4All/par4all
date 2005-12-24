/* $Id$ */

#ifndef ACCESS_DESCRIPTION_H
#define ACCESS_DESCRIPTION_H

/*
 * just something looking like Newgen Domains for homogeneity
 *
 * should use normalized somewhere, somehow...
 */

#define access			int

#define	access_undefined        (-1)
#define	local_constant		 (1)
#define local_shift		 (2)
#define local_star		 (3)
#define local_affine		 (4)
#define aligned_constant	 (5)
#define aligned_shift		 (6)
#define aligned_star		 (7)
#define aligned_affine		 (8)
#define not_aligned		 (9)
#define local_form_cst          (10)

#define	access_undefined_p(a)		(a == access_undefined)

#define access_tag(a)			(a)
#define make_access(a)			(a)

#endif

