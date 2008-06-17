      program descriptive
      integer A1(100)
chpf$ processors P(4)
chpf$ distribute A1(block) onto P
      call init(A1)
      call output(A1)
      end
      subroutine init(A)
      integer A(100)
chpf$ processors P(4)
chpf$ distribute A * (block) onto P
      integer i
chpf$ independent
      do i=1, 100
         A(i)=i
      enddo
      end
      subroutine output(A)
      integer A(100)
chpf$ processors P(4)
chpf$ distribute A * (block) onto P
      integer i
      print *, (A(i), i=45, 50)
      end
