c replication and redistributions
      program replication
      integer n, i
      parameter(n=100)
      integer a(n), b
cfcd$ setbool('HPFC_LAZY_MESSAGES',1)
cfcd$ setbool('HPFC_USE_BUFFERS', 1)
cfcd$ setbool('HPFC_GUARDED_TWINS',0)
chpf$ dynamic a
chpf$ template t0(n)
chpf$ align a(i) with t0(i)
chpf$ processors p0(10)
chpf$ distribute t0(block) onto p0
chpf$ template t1(n,n,n)
chpf$ processors p1(2,2,2)
chpf$ distribute t1(block, block, block) onto p1
chpf$ template t2(n,n)
chpf$ processors p2(5,2)
chpf$ distribute t2(cyclic, block) onto p2
chpf$ independent
      do i=1, n
         a(i) = n-i+1
      enddo
chpf$ realign a(i) with t1(*,i,*)
      b=a(7)
c
c this is the realignment we are interested in.
c
c     P1(*,B,*) -> P2(C,*)
c
chpf$ realign a(i) with t2(i,*)
      b=b+a(7)
      end
