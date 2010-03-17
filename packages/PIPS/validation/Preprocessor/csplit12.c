/* Bug found in mesa.c, but not reproduced in the small...  */

typedef unsigned int	GLuint;		 
typedef unsigned char	GLboolean;
typedef float		GLfloat;	 

typedef void (*line_func)(GLuint v1, GLuint v2, GLuint pv );

struct gl_2d_map {
	GLuint Uorder;		 
	GLuint Vorder;		 
	GLfloat u1, u2;
	GLfloat v1, v2;
	GLfloat *Points;	 
	GLboolean Retain;	 
};

main()
{
  ;
}
