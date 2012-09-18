struct s { int ** pp1; int **pp2;};
void foo(struct s *ps)
{   int i = 1, j = 2;   // S1
    *((*ps).pp1) = &i;  // S2
    *((*ps).pp2) = &j;  // S3
    (*ps).pp1 = (*ps).pp2;  // S4
    return;
}
