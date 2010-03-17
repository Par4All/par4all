      program NS
      parameter (nvar=3,nxm=2000,nym=2000)
      real phi(nvar,nxm,nym)
      ncont=0
      nx=101
      ny=101

      open(1,file='stokes.ini',status='old')
      if(ncont.eq.1) then
       do j=1,ny
          do i=1,nx
             read(1,*) x,y,(phi(iv,i,j),iv=1,nvar),divu,uno
          enddo
       enddo
      else
       do j=1,ny
          do i=1,nx
             read(1,*) x,y,(phi(iv,i,j),iv=1,nvar),divu,uno
          enddo
       enddo
      endif
      close(1)
      end
