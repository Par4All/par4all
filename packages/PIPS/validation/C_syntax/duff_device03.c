int duff_device03(char *from, char *to, int count)
{
  {
    int n = (count + 7) / 8;

    switch(count % 8) {
    case 0:
    again: *to++ = *from++;

    case 7: *to++ = *from++;
    case 6: *to++ = *from++;
    case 5: *to++ = *from++;
    case 4: *to++ = *from++;
    case 3: *to++ = *from++;
    case 2: *to++ = *from++;
    case 1: *to++ = *from++;
      if(--n > 0) goto again;
    }
  }

  return count;
}
