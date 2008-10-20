c test massive I/Os
      program hpftest49
c
c testing massive I/Os, that is scatter-gathers...
c      
      integer size
      parameter(size=500)
      integer a(size), i, n
chpf$ template t(size+10)
chpf$ processors p(4)
chpf$ align a(i) with t(i+3)
chpf$ distribute t(block) onto p
      print *, 'hpftest 49 running...'
      a(1)=17
      read *, n
chpf$ independent(i)
      do i=2, size
         a(i)=i+10
      enddo
      print *, (a(i),i=1,n)
      print *, 'that''s all'
      end
