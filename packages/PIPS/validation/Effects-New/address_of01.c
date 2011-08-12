// Check the declaration effects and the resulting use-def chains
//
// Fabien would like the is-referenced effects of "&i" be different from
// "i = ...", but what he really has in mind is a basic alias
// analysis, very much Andersen like. Funny, since he claimed that
// Andersen analysis was too simplistic to be useful:-).


void address_of01()
{
  int /* register */ i;
  int j;
  int *p;

  p = &i;
  j = 1;

  // *p cannot write j because the address of j is never copied
  *p = 2;

  return;
}
