#ifndef _param_H_
#define _param_H_
#if (defined(__STDC__) || defined(__cplusplus))

extern char **Read_ParamNames(FILE *in, int m);

#else /* (defined(__STDC__) || defined(__cplusplus)) */

extern char **Read_ParamNames(/* FILE *in, int m */);

#endif /* (defined(__STDC__) || defined(__cplusplus)) */
#endif /* _param_H_ */
