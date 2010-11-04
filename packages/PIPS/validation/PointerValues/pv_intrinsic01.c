#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

int main()
{
  FILE* fp;
  size_t n, nr;
  char buf1[200];
  char *buf2 = malloc(200 * sizeof(char));
  char * fmt;
  char * i_name;
  int i, r, c, max;
  fpos_t * fp_pos, pos;
  long int fp_pos_indic;
  va_list vl;

  // formatted IO functions
  fmt = "%s%d";
  max = 100;
  n = 200;
  fp = fopen ("file.txt", "rb");


  (void) fread (buf1, 8, 50, fp);
  (void) fscanf(fp, fmt, i_name, &i);
  (void) fprintf(fp, "%s%d", i_name, i);


  (void) scanf("%s%d",i_name, &i);
  (void) printf(fmt, i_name, i);


  (void) sscanf(buf1, fmt, i_name, &i);
  (void) sprintf(buf2, fmt, i_name, i);
  (void) snprintf(buf2, 100, fmt, i_name, i);


  // character IO functions

  c = fgetc(fp);
  (void) fgets(buf1, max, fp);
  (void) fputc(c, fp);
  (void) fputs(buf1, fp);

  c = getc(fp);
  (void)putc(c, fp);
  (void) ungetc(c, fp);

  c = getchar();
  (void) putchar(c);

  (void) gets(buf1);
  (void) puts(buf1);

  // direct IO functions

  nr = fread(buf2, sizeof(char), n, fp);
  nr = fwrite(buf2, sizeof(char), n, fp);

  // file positionning functions
  fp_pos = &pos;
  (void) fgetpos(fp, fp_pos);
  (void) fgetpos(fp, &pos);

  (void) fseek(fp, 0L, SEEK_SET);

  (void) fsetpos(fp, fp_pos);

  fp_pos_indic = ftell(fp);
  rewind(fp);

  // error handling functions

  clearerr(fp);
  r = feof(fp);
  r = ferror(fp);
  perror(buf1);

  fclose (fp);

  fprintf(stderr, "The END\n");

  return(0);
}
