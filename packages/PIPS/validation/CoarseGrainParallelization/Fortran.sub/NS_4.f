       program NS
       parameter (nvar=3,nxm=2000,nym=2000)
       real phi(nvar,nxm,nym), phi1(nvar,nxm,nym)
       real xcoef(nxm,nym)

       nx=101
       ny=101

       unorm=0.
! This loop nest should be distributed,
! the reduction on unorm output as OpenMP
! and the 2 resulting loop nests parallelized.
       do j=1,ny
          do i=1,nx
             unorm=unorm+(phi1(1,i,j)-phi(1,i,j))**2
     1            +(phi1(2,i,j)-phi(2,i,j))**2
             do iv=1,nvar
                phi(iv,i,j)=phi1(iv,i,j)
             enddo
          enddo
       enddo
       end
