// Pointer to array


// AM+FI: make sure the points-to information is typed and C compatible

// We expect here p->_p_1

int ptr_to_array01(int * (*p)[10])
{
  int a;

  (*p)[3] = &a;

  return 0;
}
