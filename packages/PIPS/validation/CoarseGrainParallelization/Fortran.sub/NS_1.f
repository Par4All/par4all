       program NS
       parameter (nvar=3,nxm=2000,nym=2000)
       real phi(nvar,nxm,nym)
       nx=101
       ny=101

!     The loop on j should be parallelized and i privatized :
       do j=1,ny
          do i=1,nx
             phi(1,i,j)=0.
             phi(2,i,j)=0.
             phi(3,i,j)=0.
          enddo
       enddo

       end
