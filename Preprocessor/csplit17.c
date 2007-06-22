/* See what happens with specific parsing for compilation units */

extern int csplit17(int i, int * pj, int k, int * * pl);

int csplit17(int i, int * pj, int k, int * * pl)
{
  *pj = i;
  **pl = k;
}
