#ifndef _UTIL_H_
#define _UTIL_H_
#ifdef __cplusplus
extern "C" 
{
#endif
  extern int arg_p(int argc, char **argv, char *opt, int *val);
  extern int arg_s(int argc, char **argv, char *opt, char *string);



#ifdef __cplusplus
}
#endif

#endif
