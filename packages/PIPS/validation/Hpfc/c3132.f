      program c3132

      integer nr, nf, ni
      parameter (nr = 160, nf = 320, ni = 8)

      dimension t31p(nf,nr,ni)
      dimension t32(nr,nf,ni)

!hpf$ dynamic t31p

!hpf$ processors s31(16)
!hpf$ distribute t31p(block,*,*) onto s31

!hpf$ processors s32(10)
!hpf$ distribute t32(block,*,*) onto s32
      
      t31p(1,1,1) = 2

!hpf$ realign t31p(i,j,k) with t32(j,i,k)

      print *, t31p(1,1,1)

      end

!!hpf$ independent
!      do i=1, nr
!!hpf$    independent
!         do j=1, nf
!!hpf$       independent
!            do k=1, ni
!               t32(i,j,k) = t31p(j,i,k)
!            enddo
!         enddo
!      enddo
!
!      print *, t32(1,1,1)
