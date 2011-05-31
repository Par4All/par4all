struct ure { int lle; float ing[2] ; };

struct ure llement()
{
    struct ure noth = { 1, {2,3} };
    noth.ing[0] = 2.;
    return noth;
}
main()
{
    struct ure nature = { 2,{3,4}};
    nature=llement();
    printf("%f",nature.lle+nature.ing[0]*nature.ing[1]);
}
