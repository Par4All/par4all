c SOR 2x2 parameter and flip-flop
      program hpftest52
c
c version avec 4 processeurs
c
      parameter (n=500)
      parameter (time=40)
      real temp(n,n,2), north(n), x
      integer old, new
CHPF$ TEMPLATE t(n,n)
CHPF$ ALIGN temp(i,j,*) with t(i,j)
CHPF$ ALIGN north(i) with t(1,i)
CHPF$ PROCESSORS p(2,2)
CHPF$ DISTRIBUTE t(block,block) ONTO p
      print *, 'HPFTEST52 RUNNING'
      print *, 'THERMO'
c
c initialization
      print *, 'INITIALIZING'
chpf$ independent(i)
      do i=1,n
         north(i) = 100.0
      enddo
      do k=1,2
chpf$ independent(i)
         do i=1,n
            temp(1,i,k) = north(i)
         enddo
chpf$ independent(j,i)
         do j=1,n
            do i=2,n
               temp(i,j,k) = 10.0
            enddo
         enddo
      enddo
c
      print *, 'RUNNING'
      new = 2
      old = 1
      do k=1,time
c
c computation and copy back (dataparallel semantic)
chpf$ independent(j,i)         
         do j=2,n-1
            do i=2,n-1
               temp(i,j,new) = 0.25*
     $              (temp(i-1,j,old) + temp(i+1,j,old) + 
     $               temp(i,j-1,old) + temp(i,j+1,old))
            enddo
         enddo
         old = new
         new = 3-new
      enddo
c
c print results
c
chpf$ independent(i)
      do i=1, n
         north(i) = temp(2,i,old)
      enddo
      x = REDMAX1(north(1), 1, n)
c
      print *, 'RESULTS:'
      print *, 'MAX', x
 10   format(F8.2, F8.2, F8.2, F8.2, F8.2)
      do i=2, 10, 2
         write (6,10) 
     $        temp(i,12,old), temp(i,24,old), temp(i,36,old),
     $        temp(i,48,old), temp(i,60,old)
      enddo
      print *, 'HPFTEST52 ENDED'
      end
      real function REDMAX1(a,l,u)
      integer l, u
      real a(l:u), amax
      amin = a(l)
      do i=l+1, u
         if (a(i).GT.amax) amax = a(i)
      enddo
      redmax1 = amax
      return
      end
