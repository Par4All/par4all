struct ure { int lle; float ing ; };

struct ure llement()
{
    struct ure noth = { 1 };
    noth.ing = 2.;
    return noth;
}
main()
{
    struct ure nature = { 2};
    nature=llement();
    printf("%f",nature.lle+nature.ing);
}
