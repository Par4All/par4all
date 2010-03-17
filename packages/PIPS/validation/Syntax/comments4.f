      subroutine comments4(t,n)
      real t(n)

c     Do we lose comments in front of an ENDIF?

C     No, but there is a bug when the user view is displayed: 
C     a useless ELSE appears

      do i = 1, N
         if(t(i).lt.0.) then
            t(i) = 0.
c     this comment might be lost!
            endif
c     And how about this one?
      enddo

      end
