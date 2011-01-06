void foo()
{
    int i;
    for(i=0;i!=5;i++) {
      if (i == 3)
	goto the_end;
        printf("%d",i);
    }
 the_end:
    printf("Exit with %d",i);
}
