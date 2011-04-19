#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

int main(char *fmt1, ...)
{
  FILE* fp;
  size_t n, nr;
  char buf1[200];
  char *buf2 = malloc(200 * sizeof(char));
  char * fmt2;
  char * i_name;
  int i, r, c, max;
  fpos_t * fp_pos, pos;
  long int fp_pos_indic;
  va_list vl;


  // formatted IO functions
  fmt2 = "%s%d";
  max = 100;
  n = 200;
  fp = fopen ("file.txt", "rb");


  (void) fread (buf1, 8, 50, fp);
  (void) fscanf(fp, fmt2,i_name, &i);
  (void) fprintf(fp, "%s%d", i_name, i);

  va_start(vl, fmt1);
  (void) vfscanf(fp, fmt1, vl);
  va_end(vl);
  va_start(vl, fmt1);
  (void) vfprintf(fp, fmt1, vl);
  va_end(vl);

  (void) scanf("%s%d",i_name, &i);
  (void) printf(fmt2, i_name, i);

  va_start(vl, fmt1);
  (void) vscanf(fmt1, vl);
  va_end(vl);
  va_start(vl, fmt1);
  (void) vprintf(fmt1, vl);
  va_end(vl);


  (void) sscanf(buf1, fmt2, i_name, &i);
  (void) sprintf(buf2, fmt2, i_name, i);
  (void) snprintf(buf2, 100, fmt2, i_name, i);

  va_start(vl, fmt1);
  (void) vsscanf(buf1, fmt1,vl);
  va_end(vl);
  va_start(vl, fmt1);
  (void) vsnprintf(buf2, 100, fmt1, vl);
  va_end(vl);
  va_start(vl, fmt1);
  (void) vsprintf(buf2, fmt1, vl);
  va_end(vl);

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
  perror(buf1); /* should also have an effect on errno */

  fclose (fp);


  fprintf(stderr, "The END\n");
}

