      program reg2
      common /t/ s1
      common /u/ s2
      integer s1, s2
      real a(1000)
      read *, s1
      read *, s2
      call sub1(a)
      end

      subroutine sub1(a)
      common /t/ s1
      common /u/ s2
      integer s1, s2
      real a(s1, s2)
      call sub2(a)
      end

      subroutine sub2(a)
      common /t/ s1
      integer s1
      real a(s1, 1)
      call sub3(a)
      end
      
      subroutine sub3(a)
      common /t/ s1
      common /u/ s2
      integer s1, s2
      integer j, k
      real a(s1, 1)
      logical q
      read *, q
      if (q) then
         do j=1, s1
            do k=1, s2
               a(j, k) = 1.0
            enddo
         enddo
      endif
      end
      
