/* Effect bug in ammp: call site to a_m_serial seems to lack argument(s) */

/* Points-to bug: unitialized pointer dereferencing... */

typedef struct atom {
  int serial;
  int active;
  struct atom * next;
} ATOM;

int atomUPDATE;
int atomNUMBER;
ATOM * first;

ATOM *a_m_serial(int serial)
{
  static ATOM *ap = ((void *)0);
  static ATOM *lastmatched = ((void *)0);
  int i , n, a_number();
  if( atomUPDATE) n= a_number();
  else n = atomNUMBER;

  ap = first; /* static pointer is hook for more efficient search */
  if( ap == ((void *)0)) return ((void *)0);
  if( lastmatched == ((void *)0) ) lastmatched = first;

  if( serial == lastmatched->serial) return lastmatched;
  if( serial > lastmatched->serial) ap = lastmatched;
  for( i=0; i< n; i++ )
    {
      if( ap-> serial == serial) {lastmatched = ap;return ap;}
      if( ap == ap->next)ap = first ;
      else ap = ap->next;
    }
  return ((void *)0);
}

int a_number()
{
  ATOM *ap;
  if( atomUPDATE )
    {
      atomUPDATE = 0;
      atomNUMBER = 0;
      if( first == ((void *)0) ) return 0 ;
      ap = first;
      while(1)
	{
	  if( ap->next == ((void *)0)) break;
	  atomNUMBER++;
	  if( ap->next == ap ) break;
	  ap = ap->next;
	}
    }
  return atomNUMBER;
}

ATOM *a_next(int flag)
{
  static ATOM *ap = ((void *)0);
  if( ap == ((void *)0)) ap = first ;
  if( ap == ((void *)0)) return ((void *)0);
  if( flag <= 0){ ap = first; return ap;}
  if( ap == ap->next) return ((void *)0);
  ap = ap->next;
  return ap;
}

int activate (int i1, int i2)
{
  int upper, lower;
  ATOM *ap,*a_m_serial(),*a_next();
  int i ,numatm,a_number();

  if( i2 == 0 )
    {
      ap = a_m_serial( i1) ;
      if( ap != ((void *)0))
	ap -> active = 1;
      return 0;
    }

  upper = i2; lower = i1;
  if( i2 < i1 ) { lower = i2; upper = i1;}

  numatm = a_number();
  for( i=0; i< numatm; i++)
    {
      ap = a_next(i);
      if( ap->serial >= lower && ap->serial <= upper)
	ap->active = 1;
    }
  return 0;
}
