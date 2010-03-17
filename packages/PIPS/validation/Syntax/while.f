      program dowhile
      
      integer i, while
      real do while
      
c      i=3
c      do , while (i.gt.0)
c         print *, i
c         i = i - 1
c      end do

      i=4
      do while (i.gt.1)
         print *, i+2
         i = i - 2
      end do

      i=5
      do 10 while (i.gt.2)
         print *, i+3
         i = i - 3
 10   continue

c      i=2
c      do 20 , while (i.gt.3)
c         print *, i+4
c         i = i - 4
c 20   continue

      do while = 1, 3
         print *, while
      enddo

      do while = 1.2
      print *, do while

      end

