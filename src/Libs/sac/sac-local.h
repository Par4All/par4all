
#ifndef __SAVC_LOCAL_H__
#define __SAVC_LOCAL_H__

typedef struct {
      int type;
      list args;        /* <expression> */
} _match, * match;

#define MATCH(x) ((match)((x).p))


typedef struct {
      char * name;
      int vectorSize;       /* nb of subwords in a register argument */
      int subwordSize;      /* nb bits per subword */
} _opcode, * opcode;

#define OPCODE(x) ((opcode)((x).p))


typedef struct {
      int nbArgs;
      list opcodes;     /* <opcode> */
} _operation, * operation;

#define OPERATION(x) ((operation)((x).p))

#endif /*__SAVC_LOCAL_H__*/
