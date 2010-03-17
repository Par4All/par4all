      program alias
      parameter (n=10)
      integer n, m
      real*4 a(n), c
      common /essai/ c(100)
      print *, 'detection des alias dynamiques'

      print *, 'pas ok'
      call sub(a, a, n, 4*n, 4*n, 4)

      print *, 'ok'
! comment connait-on la taille? regions? ou declaration...
      m = 5
      call sub(a(1), a(m+1), m, 4*m, 4*m, 4)

      print *, 'pas ok'
      m = 3
      call sub(a(1), a(m), m, 4*m, 4*m, 4)

      print *, 'pas ok'
      call sub(c(1), c(11), 10, 4*10, 4*10, 4)

      print *, 'ok'
      call sub(c(11), c(21), 10, 4*10, 4*10, 4)

      print *, 'fin'
      end

      subroutine sub(a, b, m, sa, sb, sm)
      common /essai/ c(100)
      real*4 c
      integer m, i
      real*4 a(m), b(m)
      integer sa, sb, sm

! W-MUST a(1:m) b(1:m) c(1:m)
! R a b c m
c      call checkalias('ab\0', a, min(sa,4*m), b, min(sb,4*m))
c      call checkalias('ac\0', a, min(sa,4*m), c, min(4*100,4*m))
c      call checkalias('am\0', a, min(sa,4*m), m, min(sm,4))
c      call checkalias('bc\0', b, min(sb,4*m), c, min(4*100,4*m))
c      call checkalias('bm\0', b, min(sb,4*m), m, min(sm,4))
c      call checkalias('cm\0', c, min(4*100,4*m), m, min(sm,4))

      do i=1, m
         a(i) = i
      enddo

      do i=1, m
         b(i) = a(i) + 3
      enddo

      do i=1, m
         c(i) = c(i) + b(i)
      enddo
      end








