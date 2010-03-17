      program inherit
      real A(10)
chpf$ distribute A
      call init(A)
      call affiche(A)
      end
      subroutine init(A)
      real A(10)
      integer i
chpf$ inherit A
chpf$ independent
      forall (i=1:10) A(i)=i
      end
      subroutine affiche(A)
      real A(10)
      integer i
chpf$ inherit A
      print *, (A(i), i=1, 10)
      end
