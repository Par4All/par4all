#ifndef _param_H_
#define _param_H_
#ifdef __STDC__

extern char **Read_ParamNames(FILE *in, int m);

#else /* __STDC__ */

extern char **Read_ParamNames(/* FILE *in, int m */);

#endif /* __STDC__ */
#endif /* _param_H_ */
