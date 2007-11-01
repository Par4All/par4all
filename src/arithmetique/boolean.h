#ifndef BOOLEAN_INCLUDED
#define BOOLEAN_INCLUDED

typedef enum { false, true } boolean;
#define	TRUE     true
#define	FALSE    false

#define boolean_string(b) ((b)? "TRUE" : "FALSE")

#endif
