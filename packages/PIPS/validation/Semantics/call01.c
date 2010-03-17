/* Chekc that unvisible side effects are not shown in the summary transformer */

int j = 0;

void call01(int i)
{
  i++;
  j++;
}
