int assign_dead_code(int r)
{
  // assignment dead codes
  r += 0;
  r -= 0;
  r *= 1;
  r /= 1;
  r |= 0;
  r &= r;
  r |= r;
  // r = 0;
  r ^= r;
  return r;
}
