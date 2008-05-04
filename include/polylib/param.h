#ifndef _param_H_
#define _param_H_

#if defined(__cplusplus)
extern "C" {
#endif

extern const char **Read_ParamNames(FILE *in, int m);
extern void Free_ParamNames(const char **params, int m);

#if defined(__cplusplus)
}
#endif

#endif /* _param_H_ */
