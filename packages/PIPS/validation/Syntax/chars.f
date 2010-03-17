! test character declarations
      program chars
      character*1 lookup(16)
      data lookup/'A','B','C','D','E','F','G','H',
     1            'I','J','K','L','M','N','O','P'/
      do i=1, 16
         print *, lookup(i)
      enddo
      end
