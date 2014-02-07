int duff_device02(char *from, char *to, int count)
{
  {
    int n = (count + 7) / 8;
      
    switch(count % 8) {
    case 0: 
    loop1: do { *to++ = *from++;
      case 7: *to++ = *from++;
      case 6: *to++ = *from++;
      case 5: *to++ = *from++;
      case 4: *to++ = *from++;
      case 3: *to++ = *from++;
      case 2: *to++ = *from++;
      case 1: *to++ = *from++;
      } while(--n > 0);
    }
  }

  return count;
}
