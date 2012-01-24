      program main
      implicit none
      integer k, i, s

      do 10 k=1 , 10
         s = 0
         i = 1
         do while (i .LE. 100)
            s = s + 1
            i = i + 1
         enddo
         print *, s
 10   continue
      end
