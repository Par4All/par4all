/* Effect bug in ammp: call site to a_m_serial seems to lack argument(s) */

typedef struct atom {
  int serial;
  int active;
  struct atom * next;
} ATOM;

int atomUPDATE;
int atomNUMBER;
ATOM * first;


int a_number()
{
 ATOM *ap;
 l:
 ap = first;
 if( ap->next == ((void *)0)) goto l;
 
 return atomNUMBER;
}

