/* Image format information. */
#define JAS_IMAGE_MAXFMTS 10
typedef struct  {
  int id;
  /* The ID for this format. */
  char *name;
  /* The name by which this format is identified. */
  char *ext;
  /* The file name extension associated with this format. */
  char *desc;
  /* A brief description of the format. */
  /* jas_image_fmtops_t ops; */
  /* The operations for this format. */
} jas_image_fmtinfo_t;

static jas_image_fmtinfo_t jas_image_fmtinfos[JAS_IMAGE_MAXFMTS];
void jas_image_clearfmts(int jas_image_numfmts)
{
  int i;
  jas_image_fmtinfo_t *fmtinfo;
  for (i = 0; i < jas_image_numfmts; ++i) {
    fmtinfo = &jas_image_fmtinfos[i];
    if (fmtinfo->name) {
      jas_free(fmtinfo->name);
      fmtinfo->name = 0;
    }
    if (fmtinfo->ext) {
      jas_free(fmtinfo->ext);
      fmtinfo->ext=0;
    }
    if(fmtinfo->desc){
      jas_free(fmtinfo->desc);
      fmtinfo->desc = 0;
    }
  }
  jas_image_numfmts = 0;
}
