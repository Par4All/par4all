C     Modifications
C      - suppression de l'exit de while dans la boucle 1 afin de tester
C     l'analyse semantique; ce ne serait pas necessaire si le graphe
C     de controle etait mieux construit; Francois Irigoin, 11 avril 1991
C
      program tmines
c
c***********************************************************************
c
c     programme de calcul de l'ecoulement potentiel
c           dans une tuyere a section rectangulaire variable
c
c     ce programme a ete realise a l'onera par BREDIF en utilisant
c     des disques pour des stockages intermediaires et des stockages
c     de resultats.
c     il a ete modifie par madame GUIRAUD pour n'utiliser que
c     la memoire centrale et des sorties imprimante.
c
c     le programme est autonome,il fabrique toutes ses donnees
c
c
c     les dimensions sont fixees dans la carte parameter
c
c**********************************************************************
c
      parameter (ni=81,nj=15,nk=25)
c
      dimension phi(ni,nj,nk),q(ni,nj,nk),r(ni,nj,nk),t(ni,nj,nk)
      dimension b(14,ni,nj,nk),bel(37*(ni-1)*(nj-1)),a(14,ni,nj,nk)
      dimension belk(37*(ni-1)*(nj-1),nk-1)
c
      common/calcgx/eps,nmax,resnl
      common/resulx/imp,niter,numtot
      common/mach/xmach,xmin,xmax,xnewt,idec
c
c
      nresul=6
c
c
c
      im=ni
      jm=nj
      km=nk
                         xmach=.  26
      eps=.1
      nmax=50
      nmaxnl=50
      resnl=.1e-06
      xmin=.95
      xmax=1.40
      xnewt=.95
      idec=1
c
c
c
      write(6,1000)
1000  format(1h1//1x,130('*')//20x,' programme tmines de resolution',
     $' de l equation du potentiel'//20x,' calcul de l ecoulement irr'
     $,'otationnel isentropique d un fluide compressible non visqueux'
     $//21x,'dans un convergent'//20x,' discretis'
     $,'ation elements finis avec integration de gauss(schema a 27 po'
     $,'ints)'//20x,' resolution par newton sur la densite et ',
     $'methode de gradient conjugue avec preconditionnement'//20x,
     $' par la factorisee incomplete de choleski par points'//20x,
     $' auteur marc bredif - onera version du 17/03/85'//1x,130('*')/)
c
c
c---------initialisation----------------------------------------------
c
      call mailla(im,jm,km,q,t,b,phi)
c
      call poltri
c
      call calmat(im,jm,km,r,q,t,b,bel,belk)
c
      write(6,2000) im,jm,km
2000  format(10x//30x,'test:tuyere 2./1./1.5 a raccords paraboliques',
     .30x        //20x,'imax=',i5,' jmax=',i5,' kmax=',i5/)
c
      niter=0
      numtot=0
c
      call resul(nresul,im,jm,km,phi,r)
c
c----------boucle externe---------------------------------------------
c
      do 1 niter=1,nmaxnl
c
      call romat(im,jm,km,phi,r,bel,b,a,belk)
      call calcg(im,jm,km,phi,q,r,t,b,a)
      call resul(nresul,im,jm,km,phi,r)
C     if(resnl.lt.0.) go to 2
c
1     continue
c
2     continue
c
      write(6,'(3i10)') im,jm,km
      write(6,'(5e16.7)')(((phi(i,j,k),i=1,im),j=1,jm),k=1,km)
c
 3000 stop
      end
c
      subroutine mailla(im,jm,km,x,y,z,phi)
c
      dimension x(im,jm,km),y(im,jm,km),z(im,jm,km),phi(im,jm,km)
      common/paroi/xp(81),yp(81),zp(81)
      common/mach/xmach,xmin,xmax,xnewt,xdec
c
c
c
      i2=im/2+1
      q=.5**(1./float(i2))
      xp(1)=0.
      xp(2)=1.
      do 1 i=3,i2
 1    xp(i)=xp(i-1)+q*(xp(i-1)-xp(i-2))
                         ro=.5/ xp(i2)
      if(mod(im,2).eq.0) ro=1./(xp(i2)+xp(i2-1))
      xp(i2)=xp(i2)*ro
      do 2 i=1,i2-1
      xp(i)=xp(i)*ro
 2    xp(im-i+1)=1.-xp(i)
c
      do 3 i=1,im
      s=-5.+10.*xp(i)
                   t=2.
      if(s.ge.-2.) t=2.  - .50*(s+2.)**2
      if(s.ge.-1.) t=1.  + .50*  s   **2
      if(s.ge. 0.) t=1.  + .25*  s   **2
      if(s.ge. 1.) t=1.5 - .25*(s-2.)**2
      if(s.ge. 2.) t=1.5
      xp(i)=s
      yp(i)=t
 3    zp(i)=t
c
      do 10 i=1,im
      do 10 j=1,jm
      do 10 k=1,km
10    x(i,j,k)=xp(i)
c
      do 20 i=1,im
      do 20 k=1,km
      do 20 j=1,jm
      y(i,j,k)=yp(i)*float(j-1)/float(jm-1)
20    z(i,j,k)=zp(i)*float(k-1)/float(km-1)
c
      do 100 j=1,jm
      do 100 k=1,km
100   phi(1,j,k)=0.
      do 200 i=2,im
      srap=yp(im)*zp(im)*4./(yp(i-1)+yp(i))/(zp(i-1)+zp(i))
      a=srap*xmach/((5.+xmach*xmach)**3)
      xml=.01
      do 201 l=1,100
201   xml=a*(xml*xml+5.)**3
      dphidx=sqrt((5.+xmach*xmach)/(5.+xml*xml))*xml/xmach
      do 210 j=1,jm

      do 210 k=1,km
210   phi(i,j,k)=phi(i-1,j,k)+(xp(i)-xp(i-1))*dphidx
200   continue
c
      return
      end
      subroutine calmat(im,jm,km,r,cx,cy,cz,bel,belk)
c
      dimension cx(im,jm,km),cy(im,jm,km),cz(im,jm,km),r(im,jm,km)
      dimension bel(37,im-1,jm-1)
      dimension belk(37,im-1,jm-1,km-1)
c
      common/polxyz/pxx(8,8,9),pyy(8,8,9),pzz(8,8,9),pxy(8,8,9)
     $             ,pxz(8,8,9),pyz(8,8,9),p(8,9,3),w(9)
      common/mach/xmach,xmin,xmax,xnewt,idec
      common/vect/x(8,3,81),a(3,3,81),surf(81),temp(81),res(81,8)
     $,s11(81),s12(81),s13(81),s22(81),s23(81),s33(81)
      common/secm/secm1(25,25),secm2(25,25)
c
c
c
      imm=im-1
      jmm=jm-1
      kmm=km-1
c
      do 101 k=1,km
      do 101 j=1,jm
      do 101 i=1,im
101   r(i,j,k)=0.
c
c
c
c--------boucle sur tous les elements---------------------------------
c
      do 1 k=1,kmm
c
      do 11 lb=1,37
      do 11 j=1,jmm
      do 11 i=1,imm
11    bel(lb,i,j)=0.
c
      do 10 j=1,jmm
c
      do 20 i=1,imm
      x(1,1,i)=cx(i,j,k)
      x(1,2,i)=cy(i,j,k)
      x(1,3,i)=cz(i,j,k)
      x(2,1,i)=cx(i+1,j,k)
      x(2,2,i)=cy(i+1,j,k)
      x(2,3,i)=cz(i+1,j,k)
      x(3,1,i)=cx(i+1,j+1,k)
      x(3,2,i)=cy(i+1,j+1,k)
      x(3,3,i)=cz(i+1,j+1,k)
      x(4,1,i)=cx(i,j+1,k)
      x(4,2,i)=cy(i,j+1,k)
      x(4,3,i)=cz(i,j+1,k)
      x(5,1,i)=cx(i,j,k+1)
      x(5,2,i)=cy(i,j,k+1)
      x(5,3,i)=cz(i,j,k+1)
      x(6,1,i)=cx(i+1,j,k+1)
      x(6,2,i)=cy(i+1,j,k+1)
      x(6,3,i)=cz(i+1,j,k+1)
      x(7,1,i)=cx(i+1,j+1,k+1)
      x(7,2,i)=cy(i+1,j+1,k+1)
      x(7,3,i)=cz(i+1,j+1,k+1)
      x(8,1,i)=cx(i,j+1,k+1)
      x(8,2,i)=cy(i,j+1,k+1)
      x(8,3,i)=cz(i,j+1,k+1)
20    continue
c
      do 30 l=1,8
c
      do 31 i=1,imm
31    surf(i)=w(l)
c
      do 2 lj=1,3
      do 2 lk=1,3
      do 202 i=1,imm
202   a(lk,lj,i)=0.
      do 2 ll=1,8
      do 203 i=1,imm
203   a(lk,lj,i)=a(lk,lj,i)+x(ll,lk,i)*p(ll,l,lj)
2     continue
c
      do 3 ll=1,3
      do 204 i=1,imm
204   temp(i)=a(ll,ll,i)
      do 205 i=1,imm
      surf(i)=surf(i)*temp(i)
205   a(ll,ll,i)=1.
      do 4 lj=1,3
      do 4 i=1,imm
4     a(ll,lj,i)=a(ll,lj,i)/temp(i)
      do 5 kk=1,2
      kkk=1+mod(kk-1+ll,3)
      do 215 i=1,imm
215   temp(i)=a(kkk,ll,i)
      do 206 i=1,imm
206   a(kkk,ll,i)=0.
      do 6 lj=1,3

      do 6 i=1,imm
6     a(kkk,lj,i)=a(kkk,lj,i)-temp(i)*a(ll,lj,i)
5     continue
3     continue
c
      do 207 i=1,imm
      surf(i)=abs(surf(i))
      bel(37,i,j)=bel(37,i,j)+surf(i)
      s11(i)=a(1,1,i)*a(1,1,i)+a(1,2,i)*a(1,2,i)+a(1,3,i)*a(1,3,i)
      s22(i)=a(2,1,i)*a(2,1,i)+a(2,2,i)*a(2,2,i)+a(2,3,i)*a(2,3,i)
      s33(i)=a(3,1,i)*a(3,1,i)+a(3,2,i)*a(3,2,i)+a(3,3,i)*a(3,3,i)
      s12(i)=a(1,1,i)*a(2,1,i)+a(1,2,i)*a(2,2,i)+a(1,3,i)*a(2,3,i)
      s13(i)=a(1,1,i)*a(3,1,i)+a(1,2,i)*a(3,2,i)+a(1,3,i)*a(3,3,i)
207   s23(i)=a(2,1,i)*a(3,1,i)+a(2,2,i)*a(3,2,i)+a(2,3,i)*a(3,3,i)
c
      ijl=0
      do 8 il=1,8
      do 8 jl=il,8
      ijl=ijl+1
      do 8 i=1,imm
      bel(ijl,i,j)=bel(ijl,i,j)+surf(i)*(s11(i)*pxx(il,jl,l)+s22(i)
     $*pyy(il,jl,l)+s33(i)*pzz(il,jl,l)+s12(i)*pxy(il,jl,l)+
     $s13(i)*pxz(il,jl,l)+s23(i)*pyz(il,jl,l))
8     continue
c
30    continue
c
      do 39 l=1,8
      do 39 i=1,imm
39    res(i,l)=0.
      ind=0
      do 40 il=1,7
      ind=ind+1
      jj=il+1
      do 41 i=1,imm
41    res(i,il)=res(i,il)+bel(ind,i,j)*x(il,1,i)
      do 40 jl=jj,8
      ind=ind+1
      do 43 i=1,imm
43    res(i,il)=res(i,il)+bel(ind,i,j)*x(jl,1,i)
      do 44 i=1,imm
44    res(i,jl)=res(i,jl)+bel(ind,i,j)*x(il,1,i)
40    continue
      do 45 i=1,imm
45    res(i,8)=res(i,8)+bel(36,i,j)*x(8,1,i)
c

      do 46 i=1,imm
      r(i+1,j,k)=r(i+1,j,k)-res(i,2)
      r(i+1,j,k+1)=r(i+1,j,k+1)-res(i,6)
      r(i+1,j+1,k)=r(i+1,j+1,k)-res(i,3)
      r(i+1,j+1,k+1)=r(i+1,j+1,k+1)-res(i,7)
      r(i,j,k)=r(i,j,k)-res(i,1)
      r(i,j,k+1)=r(i,j,k+1)-res(i,5)
      r(i,j+1,k)=r(i,j+1,k)-res(i,4)
46    r(i,j+1,k+1)=r(i,j+1,k+1)-res(i,8)
c
10    continue
c
c
c
      do 66 lb=1,37
      do 66 i=1,imm
      do 66 j=1,jmm
      belk(lb,i,j,k)=bel(lb,i,j)
 66   continue
c
c
1     continue
c
      do 7 j=1,jm
      do 7 k=1,km
      secm1(j,k)=-r(1,j,k)
7     secm2(j,k)=r(im,j,k)
c
      return
      end
      subroutine poltri
c
      common/polxyz/pxx(8,8,9),pyy(8,8,9),pzz(8,8,9),
     $              pxy(8,8,9),pxz(8,8,9),pyz(8,8,9),p(8,9,3),w(9)
      dimension x(9),y(9),z(9)
c
      a=sqrt(1./3.)
      ap=.5*(1.+a)
      am=.5*(1.-a)
      x(1)=am
      x(2)=am
      x(3)=ap
      x(4)=ap
      x(5)=am
      x(6)=am
      x(7)=ap
      x(8)=ap
      x(9)=.5
      y(1)=ap
      y(2)=am
      y(3)=am
      y(4)=ap
      y(5)=ap
      y(6)=am
      y(7)=am
      y(8)=ap
      y(9)=.5
      z(1)=am
      z(2)=am
      z(3)=am
      z(4)=am
      z(5)=ap
      z(6)=ap
      z(7)=ap
      z(8)=ap
      z(9)=.5

      do 111 i=1,8
111   w(i)=1./8.
      w(9)=0.
c

      do 1 i=1,9
      x1=x(i)
      x2=1.-x1
      y1=y(i)
      y2=1.-y1
      z1=z(i)
      z2=1.-z1
      p(1,i,1)=-y2*z2
      p(2,i,1)=y2*z2
      p(3,i,1)=y1*z2
      p(4,i,1)=-y1*z2
      p(5,i,1)=-y2*z1
      p(6,i,1)=y2*z1
      p(7,i,1)=y1*z1
      p(8,i,1)=-y1*z1
      p(1,i,2)=-x2*z2
      p(2,i,2)=-x1*z2
      p(3,i,2)=x1*z2
      p(4,i,2)=x2*z2
      p(5,i,2)=-x2*z1
      p(6,i,2)=-x1*z1
      p(7,i,2)=x1*z1
      p(8,i,2)=x2*z1
      p(1,i,3)=-x2*y2
      p(2,i,3)=-x1*y2
      p(3,i,3)=-x1*y1
      p(4,i,3)=-x2*y1
      p(5,i,3)=x2*y2
      p(6,i,3)=x1*y2
      p(7,i,3)=x1*y1
      p(8,i,3)=x2*y1
1     continue
c
      do 2 i=1,8
      do 2 j=1,8


      do 2 k=1,9
      pxx(i,j,k)=p(i,k,1)*p(j,k,1)
      pyy(i,j,k)=p(i,k,2)*p(j,k,2)
      pzz(i,j,k)=p(i,k,3)*p(j,k,3)
      pxy(i,j,k)=p(i,k,1)*p(j,k,2)+p(i,k,2)*p(j,k,1)
      pxz(i,j,k)=p(i,k,1)*p(j,k,3)+p(i,k,3)*p(j,k,1)
2     pyz(i,j,k)=p(i,k,2)*p(j,k,3)+p(i,k,3)*p(j,k,2)
c
      return
      end
      subroutine romat(im,jm,km,phi,r,bel,b,a,belk)
c
      dimension phi(im,jm,km),r(im,jm,km),bel(37,im-1,jm-1)
      dimension b(14,im,jm,km),a(14,im,jm,km)
      dimension roa(81,25),vi(96),vj(96),vk(96)
      dimension belk(37,im-1,jm-1,km-1)
c
      common/mach/xmach,xmin,xmax,xnewt,idec
      common/vect/ph(81,8),q2(81),rol(81),rn(81),
     $res(81,8),bid(30,81)
      common/secm/secm1(25,25),secm2(25,25)
c
c
c
      imm=im-1
      jmm=jm-1
      kmm=km-1
      xm2=xmach*xmach
      xro1=1.+.2*xm2
      xro2=-.2*xm2
      xc0=xmin*xmin
      xc1=xmax*xmax
      xc2=.5*(xc0+xc1)
      xc3=(xc1-xc0)*sqrt(.5*(xc1-xc0))
      q2max=9.*xro1/xm2/2.8
      xne2=amin1(xnewt*xnewt,xc0)
      xn2=xne2/(1.-xne2)
      do 1234 i=1,im
      do 1234 j=1,jm
1234  roa(i,j)=1.
      do 1235 i=1,imm
      vi(i)=1.
      vj(i)=0.
      vk(i)=0.
1235  continue
c
      do 101 k=1,km
      do 101 j=1,jm
      do 102 i=1,imm
102   r(i,j,k)=0.
101   r(im,j,k)=secm2(j,k)
c
      big=1.e+30
      do 100 j=1,jm
      do 100 k=1,km
      do 103 i=1,im
      do 103 l=1,14
103   b(l,i,j,k)=0.
100   b(1,1,j,k)=big
c
      do 1 k=1,kmm
c
c
      do 66 lb=1,37
      do 66 i=1,imm
      do 66 j=1,jmm
      bel(lb,i,j)=belk(lb,i,j,k)
 66   continue
c
      do 2 j=1,jmm
c
      do 3 i=1,imm
c
      ph(i,1)=phi(i,j,k)
      ph(i,2)=phi(i+1,j,k)
      ph(i,3)=phi(i+1,j+1,k)
      ph(i,4)=phi(i,j+1,k)
      ph(i,5)=phi(i,j,k+1)
      ph(i,6)=phi(i+1,j,k+1)
      ph(i,7)=phi(i+1,j+1,k+1)
      ph(i,8)=phi(i,j+1,k+1)
3     q2(i)=0.
      do 31 l=1,8
      do 31 i=1,imm
31    res(i,l)=0.
c
c
      ind=0
      do 11 il=1,7
      ind=ind+1
      jj=il+1
      do 21 i=1,imm
21    res(i,il)=res(i,il)+bel(ind,i,j)*ph(i,il)
      do 11 jl=jj,8
      ind=ind+1
      do 22 i=1,imm
22    res(i,il)=res(i,il)+bel(ind,i,j)*ph(i,jl)
      do 32 i=1,imm
32    res(i,jl)=res(i,jl)+bel(ind,i,j)*ph(i,il)
11    continue
      do 23 i=1,imm
23    res(i,8)=res(i,8)+bel(36,i,j)*ph(i,8)
      if (idec.ne.0) then
      do 34 i=1,imm
      vi(i)=amax1(res(i,2)+res(i,3)+res(i,6)+res(i,7),.1e-10)
      vj(i)=amax1(res(i,3)+res(i,4)+res(i,7)+res(i,8),0.)
34    vk(i)=amax1(res(i,5)+res(i,6)+res(i,7)+res(i,8),0.)
      endif
c
      do 33 l=1,8
      do 33 i=1,imm
33    q2(i)=q2(i)+res(i,l)*ph(i,l)
      do 24 i=1,imm
      q2(i)=amin1(q2(i)/bel(37,i,j),q2max)
24    rol(i)=xro1+xro2*q2(i)
      do 28 i=1,imm
28    rn(i)=amin1(xm2/rol(i),amax1(0.,xn2*(1./q2(i)-xm2/rol(i))))/
     $bel(37,i,j)
      do 25 i=1,imm
25    q2(i)=xm2*q2(i)/rol(i)
      do 26 i=1,imm
26    rol(i)=rol(i)*rol(i)*sqrt(rol(i))
c
      ind=0
      do 29 il=1,8
      do 29 jl=il,8
      ind=ind+1
      do 29 i=1,imm
29    bel(ind,i,j)=bel(ind,i,j)-rn(i)*res(i,il)*res(i,jl)
c
      xa2=xm2
c
      do 27 i=1,imm
c
      xa2d=xa2
      xa2=q2(i)
      ro=rol(i)
      xma2=.7*xa2+.3*xa2d
      if(xma2.lt.xc0) go to 12
      if(xma2.gt.xc1) go to 13
      if(xma2.gt.xc2) go to 14
      beta=xma2-xc0
      xmu=sqrt(beta)*beta/xc3
      go to 15
14    beta=xc1-xma2
      xmu=1.-sqrt(beta)*beta/xc3
      go to 15
13    xmu=1.
15    rod=(vi(i)*roa(i,j+1)+vj(i)*roa(i+1,j)+vk(i)*roa(i+1,j+1))/
     $(vi(i)+vj(i)+vk(i))
      roij=ro+xmu*(rod-ro)
      roa(i+1,j+1)=.5*(ro+roij)
      go to 16
12    roa(i+1,j+1)=ro
      roij=ro
16    continue
27    rol(i)=roij
c
      do 91 i=1,imm
      b(1,i,j,k)=b(1,i,j,k)+rol(i)*bel(1,i,j)
      b(4,i,j,k)=b(4,i,j,k)+rol(i)*bel(2,i,j)
      b(5,i,j,k)=b(5,i,j,k)+rol(i)*bel(3,i,j)
      b(2,i,j,k)=b(2,i,j,k)+rol(i)*bel(4,i,j)
      b(8,i,j,k)=b(8,i,j,k)+rol(i)*bel(5,i,j)
      b(10,i,j,k)=b(10,i,j,k)+rol(i)*bel(6,i,j)
      b(9,i,j,k)=b(9,i,j,k)+rol(i)*bel(8,i,j)
91    continue

      do 92 i=1,imm
      b(3,i,j+1,k)=b(3,i,j+1,k)+rol(i)*bel(11,i,j)
      b(4,i,j+1,k)=b(4,i,j+1,k)+rol(i)*bel(17,i,j)
      b(1,i,j+1,k)=b(1,i,j+1,k)+rol(i)*bel(22,i,j)
      b(7,i,j+1,k)=b(7,i,j+1,k)+rol(i)*bel(23,i,j)
      b(10,i,j+1,k)=b(10,i,j+1,k)+rol(i)*bel(25,i,j)
      b(8,i,j+1,k)=b(8,i,j+1,k)+rol(i)*bel(26,i,j)
      b(1,i,j,k+1)=b(1,i,j,k+1)+rol(i)*bel(27,i,j)
92    continue

      do 93 i=1,imm
      b(4,i,j,k+1)=b(4,i,j,k+1)+rol(i)*bel(28,i,j)
      b(5,i,j,k+1)=b(5,i,j,k+1)+rol(i)*bel(29,i,j)
      b(2,i,j,k+1)=b(2,i,j,k+1)+rol(i)*bel(30,i,j)
      b(3,i,j+1,k+1)=b(3,i,j+1,k+1)+rol(i)*bel(33,i,j)
      b(4,i,j+1,k+1)=b(4,i,j+1,k+1)+rol(i)*bel(35,i,j)
      b(1,i,j+1,k+1)=b(1,i,j+1,k+1)+rol(i)*bel(36,i,j)
      b(14,i,j,k)=b(14,i,j,k)+rol(i)*bel(7,i,j)
      b(13,i,j+1,k)=b(13,i,j+1,k)+rol(i)*bel(24,i,j)
93    continue

      do 99 i=1,imm
      b(1,i+1,j,k)=b(1,i+1,j,k)+rol(i)*bel(9,i,j)
      b(2,i+1,j,k)=b(2,i+1,j,k)+rol(i)*bel(10,i,j)
      b(7,i+1,j+1,k)=b(7,i+1,j+1,k)+rol(i)*bel(19,i,j)
      b(8,i+1,j+1,k)=b(8,i+1,j+1,k)+rol(i)*bel(20,i,j)
      b(6,i+1,j+1,k)=b(6,i+1,j+1,k)+rol(i)*bel(21,i,j)
      b(6,i+1,j,k)=b(6,i+1,j,k)+rol(i)*bel(12,i,j)
      b(8,i+1,j,k)=b(8,i+1,j,k)+rol(i)*bel(13,i,j)
      b(9,i+1,j,k)=b(9,i+1,j,k)+rol(i)*bel(14,i,j)
      b(1,i+1,j+1,k)=b(1,i+1,j+1,k)+rol(i)*bel(16,i,j)
      b(1,i+1,j,k+1)=b(1,i+1,j,k+1)+rol(i)*bel(31,i,j)
      b(2,i+1,j,k+1)=b(2,i+1,j,k+1)+rol(i)*bel(32,i,j)
      b(1,i+1,j+1,k+1)=b(1,i+1,j+1,k+1)+rol(i)*bel(34,i,j)
      b(12,i+1,j,k)=b(12,i+1,j,k)+rol(i)*bel(15,i,j)
      b(11,i+1,j+1,k)=b(11,i+1,j+1,k)+rol(i)*bel(18,i,j)
99    continue
c

      do 50 i=1,imm
      r(i+1,j,k)=r(i+1,j,k)+res(i,2)*rol(i)
      r(i+1,j,k+1)=r(i+1,j,k+1)+res(i,6)*rol(i)
      r(i+1,j+1,k)=r(i+1,j+1,k)+res(i,3)*rol(i)
      r(i+1,j+1,k+1)=r(i+1,j+1,k+1)+res(i,7)*rol(i)
      r(i,j,k)=r(i,j,k)+res(i,1)*rol(i)
      r(i,j,k+1)=r(i,j,k+1)+res(i,5)*rol(i)
      r(i,j+1,k)=r(i,j+1,k)+res(i,4)*rol(i)
50    r(i,j+1,k+1)=r(i,j+1,k+1)+res(i,8)*rol(i)
2     continue
c
1     continue
c
      call prepcg(im,jm,km,b,a)
c
c
      return
      end
      subroutine resul(nfic,im,jm,km,phi,r)
c
      dimension phi(im,jm,km),r(im,jm,km)
      common/paroi/xp(81),yp(81),zp(81)
      common/resulx/imp,niter,numtot
      dimension xm(4),ds(4),dphi(4)
      common/mach/xmach,xmin,xmax,xnewt,idec
c
c
c
      imm=im-1
      jmm=jm-1
      xm2=xmach*xmach
      xro1=1.+.2*xm2
      xro2=-.2*xm2
c
      write(6,2000)niter,numtot
      write(6,3000)
c
c
      qmax=sqrt(9.*xro1/xm2/2.8)
      do 100 i=1,imm
      dx=xp(i+1)-xp(i)
      dphi(1)=phi(i+1,1,1)-phi(i,1,1)
      dphi(2)=phi(i+1,jm,1)-phi(i,jm,1)
      dphi(3)=phi(i+1,jm,km)-phi(i,jm,km)
      dphi(4)=phi(i+1,1,km)-phi(i,1,km)
      dy=yp(i+1)-yp(i)
      dz=zp(i+1)-zp(i)
      ds(1)=dx
      ds(2)=sqrt(dx*dx+dy*dy)
      ds(3)=sqrt(dx*dx+dy*dy+dz*dz)
      ds(4)=sqrt(dx*dx+dz*dz)

      do 10 l=1,4
      q=amin1(abs(dphi(l)/ds(l)),qmax)
      ro=xro1+xro2*q*q
10    xm(l)=q*xmach/sqrt(ro)
      xpi=.5*(xp(i)+xp(i+1))
      ypi=.5*(yp(i)+yp(i+1))
      zpi=.5*(zp(i)+zp(i+1))
      write(6,1000) xpi,ypi,zpi,xm(1),xm(2),xm(4),xm(3)
100   continue
c
1000  format(10x,7f12.5)
2000  format(1h1/30x,' iteration numero ',i6,10x,i6//)
3000  format(10x,' valeurs du mach en differents points'//
     $19x,'x         y           z        mach axe     mach horiz',
     $'   mach vert   mach coin'/)
c
      return
      end
      subroutine prod(im,jm,km,q,t,b)
c
      dimension q(im,jm,km),t(im,jm,km),b(14,im,jm,km)
c
c
c
c
      imm=im-1
      do 1 k=1,km
      ku=min0(k+1,km)
c
      do 2 j=1,jm
      jl=max0(j-1,1)
      ju=min0(j+1,jm)
c

      do 10 i=1,im
      t(i,j,k)=t(i,j,k)+b(1,i,j,k)*q(i,j,k)+b(2,i,j,k)*q(i,ju,k)+
     $b(7,i,j,k)*q(i,jl,ku)+b(8,i,j,k)*q(i,j,ku)+b(9,i,j,k)*
     $q(i,ju,ku)
      t(i,ju,k)=t(i,ju,k)+b(2,i,j,k)*q(i,j,k)
      t(i,jl,ku)=t(i,jl,ku)+b(7,i,j,k)*q(i,j,k)
      t(i,j,ku)=t(i,j,ku)+b(8,i,j,k)*q(i,j,k)
10    t(i,ju,ku)=t(i,ju,ku)+b(9,i,j,k)*q(i,j,k)
c


      do 11 i=1,imm
      t(i+1,jl,k)=t(i+1,jl,k)+b(3,i,j,k)*q(i,j,k)
      t(i+1,j,k)=t(i+1,j,k)+b(4,i,j,k)*q(i,j,k)
      t(i+1,ju,k)=t(i+1,ju,k)+b(5,i,j,k)*q(i,j,k)
      t(i+1,j,ku)=t(i+1,j,ku)+b(10,i,j,k)*q(i,j,k)
      t(i+1,jl,ku)=t(i+1,jl,ku)+b(13,i,j,k)*q(i,j,k)
      t(i+1,ju,ku)=t(i+1,ju,ku)+b(14,i,j,k)*q(i,j,k)
      t(i,j,k)=t(i,j,k)+b(3,i,j,k)*q(i+1,jl,k)+b(4,i,j,k)*q(i+1,j,k)+
     $b(5,i,j,k)*q(i+1,ju,k)+b(10,i,j,k)*q(i+1,j,ku)
     $+b(13,i,j,k)*q(i+1,jl,ku)+b(14,i,j,k)*q(i+1,ju,ku)
11    continue
c

      do 12 i=2,im
      t(i,j,k)=t(i,j,k)+b(6,i,j,k)*q(i-1,j,ku)+
     $b(11,i,j,k)*q(i-1,jl,ku)+b(12,i,j,k)*q(i-1,ju,ku)
      t(i-1,jl,ku)=t(i-1,jl,ku)+b(11,i,j,k)*q(i,j,k)
      t(i-1,ju,ku)=t(i-1,ju,ku)+b(12,i,j,k)*q(i,j,k)
12    t(i-1,j,ku)=t(i-1,j,ku)+b(6,i,j,k)*q(i,j,k)
c
2     continue
1     continue
c
      return
      end
      subroutine calcg(im,jm,km,phi,q,r,t,b,a)
c
      dimension phi(*),t(*),q(*),r(*),b(*),a(*)
c
      common/calcgx/eps,nmax,resnl
      common/resulx/imp,niter,numtot
c
c
c
      rtn=1.
      ro=0.
      num=0
      npmax=im*jm*km
      do 5 l=1,npmax
      t(l)=0.
5     q(l)=0.
c
      do 1 nitl=1,nmax
c
      num=num+1
      numtot=numtot+1
c
      do 21 l=1,npmax
      r(l)=r(l)-ro*t(l)
21    t(l)=r(l)
c
      call des(im,jm,km,t,a)
      call rep(im,jm,km,t,a)
c
      rt=0.
      do 30 l=1,npmax
30    rt=rt+r(l)*t(l)
      rtd=rtn
      rtn=rt
      dq=rtn/rtd
      if(num.ne.1) go to 3
      rt0=rt*eps*eps
      if(resnl.lt.sqrt(rt/float(npmax))) go to 3
      resnl=-1.
      return
3     rts=sqrt(rt/npmax)
      write(6,1000) num,rts
1000  format(i6,e12.4)
c
      do 50 l=1,npmax
      q(l)=t(l)+dq*q(l)
50    t(l)=0.
c
      call prod(im,jm,km,q,t,b)
c
      qt=0.
      do 41 l=1,npmax
41    qt=qt+q(l)*t(l)
      ro=rt/qt
c
      do 51 l=1,npmax
51    phi(l)=phi(l)-ro*q(l)
c
      if(rt.lt.rt0) return
1     continue
c
      return
      end
      subroutine prepcg(im,jm,km,a,b)
c
      dimension b(14,im,jm,km),a(14,im,jm,km)
      dimension h2(25),h3(25),h4(25),h5(25),h6(25),h7(25)
      dimension h8(25),h9(25),h10(25),h11(25),h12(25),h13(25),h14(25)
c
c
c
c
      jmm=jm-1
c
      do 555 l=1,14
      do 555 k=1,km
      do 555 j=1,jm
      do 555 i=1,im
555   b(l,i,j,k)=a(l,i,j,k)
      do 1 k=1,km
c
      ku=min0(k+1,km)
c
      do 20 i=1,im
c
      il=max0(1,i-1)
      iu=min0(im,i+1)
c
      do 2 j=1,jm
      dd=1./b(1,i,j,k)
      if(dd.lt.0.) then
      write(6,5555) i,j,k,dd
5555  format(3i8,'pivot = ',e15.7)
      dd=0.
      endif
      dd=sqrt(dd)
      h2(j)=b(2,i,j,k)*dd
      h3(j)=b(3,i,j,k)*dd
      h4(j)=b(4,i,j,k)*dd
      h5(j)=b(5,i,j,k)*dd
      h6(j)=b(6,i,j,k)*dd
      h7(j)=b(7,i,j,k)*dd
      h8(j)=b(8,i,j,k)*dd
      h9(j)=b(9,i,j,k)*dd
      h10(j)=b(10,i,j,k)*dd
      h11(j)=b(11,i,j,k)*dd
      h12(j)=b(12,i,j,k)*dd
      h13(j)=b(13,i,j,k)*dd
      h14(j)=b(14,i,j,k)*dd
      b(1,i,j,k)=dd
      if(j.eq.jm) go to 2
      b(1,i,j+1,k)=b(1,i,j+1,k)-h2(j)*h2(j)
      b(3,i,j+1,k)=b(3,i,j+1,k)-h2(j)*h4(j)
      b(4,i,j+1,k)=b(4,i,j+1,k)-h2(j)*h5(j)
      b(7,i,j+1,k)=b(7,i,j+1,k)-h2(j)*h8(j)
      b(8,i,j+1,k)=b(8,i,j+1,k)-h2(j)*h9(j)
      b(11,i,j+1,k)=b(11,i,j+1,k)-h2(j)*h6(j)
      b(13,i,j+1,k)=b(13,i,j+1,k)-h2(j)*h10(j)
      b(6,i,j+1,k)=b(6,i,j+1,k)-h2(j)*h12(j)
      b(10,i,j+1,k)=b(10,i,j+1,k)-h2(j)*h14(j)
2     continue
c

      do 3 j=1,jm
      b(1,iu,j,k)=b(1,iu,j,k)-h4(j)*h4(j)
      b(1,il,j,ku)=b(1,il,j,ku)-h6(j)*h6(j)
      b(1,i,j,ku)=b(1,i,j,ku)-h8(j)*h8(j)
      b(1,iu,j,ku)=b(1,iu,j,ku)-h10(j)*h10(j)
      b(2,iu,j,k)=b(2,iu,j,k)-h4(j)*h5(j)
      b(6,iu,j,k)=b(6,iu,j,k)-h4(j)*h8(j)
      b(8,iu,j,k)=b(8,iu,j,k)-h4(j)*h10(j)
      b(11,iu,j,k)=b(11,iu,j,k)-h4(j)*h7(j)
      b(12,iu,j,k)=b(12,iu,j,k)-h4(j)*h9(j)
      b(7,iu,j,k)=b(7,iu,j,k)-h4(j)*h13(j)
      b(9,iu,j,k)=b(9,iu,j,k)-h4(j)*h14(j)
      b(3,il,j,ku)=b(3,il,j,ku)-h6(j)*h7(j)
      b(4,il,j,ku)=b(4,il,j,ku)-h6(j)*h8(j)
      b(5,il,j,ku)=b(5,il,j,ku)-h6(j)*h9(j)
      b(2,il,j,ku)=b(2,il,j,ku)-h6(j)*h12(j)
      b(2,i,j,ku)=b(2,i,j,ku)-h8(j)*h9(j)
      b(4,i,j,ku)=b(4,i,j,ku)-h8(j)*h10(j)
      b(3,i,j,ku)=b(3,i,j,ku)-h8(j)*h13(j)
      b(5,i,j,ku)=b(5,i,j,ku)-h8(j)*h14(j)
      b(2,iu,j,ku)=b(2,iu,j,ku)-h10(j)*h14(j)
3     continue
c

      do 4 j=1,jm
      b(2,i,j,k)=h2(j)
      b(3,i,j,k)=h3(j)
      b(4,i,j,k)=h4(j)
      b(5,i,j,k)=h5(j)
      b(6,i,j,k)=h6(j)
      b(7,i,j,k)=h7(j)
      b(8,i,j,k)=h8(j)
      b(9,i,j,k)=h9(j)
      b(10,i,j,k)=h10(j)
      b(11,i,j,k)=h11(j)
      b(12,i,j,k)=h12(j)
      b(13,i,j,k)=h13(j)
      b(14,i,j,k)=h14(j)
4     continue
c

      do 5 j=2,jm
      b(1,iu,j-1,k)=b(1,iu,j-1,k)-h3(j)*h3(j)
      b(1,i,j-1,ku)=b(1,i,j-1,ku)-h7(j)*h7(j)
      b(1,il,j-1,ku)=b(1,il,j-1,ku)-h11(j)*h11(j)
      b(1,iu,j-1,ku)=b(1,iu,j-1,ku)-h13(j)*h13(j)
      b(2,iu,j-1,k)=b(2,iu,j-1,k)-h3(j)*h4(j)
      b(6,iu,j-1,k)=b(6,iu,j-1,k)-h3(j)*h7(j)
      b(9,iu,j-1,k)=b(9,iu,j-1,k)-h3(j)*h10(j)
      b(8,iu,j-1,k)=b(8,iu,j-1,k)-h3(j)*h13(j)
      b(12,iu,j-1,k)=b(12,iu,j-1,k)-h3(j)*h8(j)
      b(2,il,j-1,ku)=b(2,il,j-1,ku)-h6(j)*h11(j)
      b(2,i,j-1,ku)=b(2,i,j-1,ku)-h7(j)*h8(j)
      b(5,i,j-1,ku)=b(5,i,j-1,ku)-h7(j)*h10(j)
      b(4,il,j-1,ku)=b(4,il,j-1,ku)-h7(j)*h11(j)
      b(4,i,j-1,ku)=b(4,i,j-1,ku)-h7(j)*h13(j)
      b(5,il,j-1,ku)=b(5,il,j-1,ku)-h8(j)*h11(j)
      b(2,iu,j-1,ku)=b(2,iu,j-1,ku)-h10(j)*h13(j)
5     continue
c

      do 6 j=1,jmm
      b(1,iu,j+1,k)=b(1,iu,j+1,k)-h5(j)*h5(j)
      b(1,i,j+1,ku)=b(1,i,j+1,ku)-h9(j)*h9(j)
      b(1,il,j+1,ku)=b(1,il,j+1,ku)-h12(j)*h12(j)
      b(1,iu,j+1,ku)=b(1,iu,j+1,ku)-h14(j)*h14(j)
      b(6,iu,j+1,k)=b(6,iu,j+1,k)-h5(j)*h9(j)
      b(7,iu,j+1,k)=b(7,iu,j+1,k)-h5(j)*h10(j)
      b(11,iu,j+1,k)=b(11,iu,j+1,k)-h5(j)*h8(j)
      b(8,iu,j+1,k)=b(8,iu,j+1,k)-h5(j)*h14(j)
      b(3,il,j+1,ku)=b(3,il,j+1,ku)-h8(j)*h12(j)
      b(3,i,j+1,ku)=b(3,i,j+1,ku)-h9(j)*h10(j)
      b(4,il,j+1,ku)=b(4,il,j+1,ku)-h9(j)*h12(j)
      b(4,i,j+1,ku)=b(4,i,j+1,ku)-h9(j)*h14(j)
6     continue
c
20    continue
c
1     continue
c
      return
      end
      subroutine des(im,jm,km,t,b)
c
      dimension t(im,jm,km),b(14,im,jm,km),tik(25)
c
c
c
      jmm=jm-1
      kmm=km-1
c
c----------------descente---------------------------------------------
c
      do 100 k=1,km
c
      ku=min0(km,k+1)
      do 1 i=1,im
c
      il=max0(i-1,1)
      iu=min0(i+1,im)
c

      do 10 j=1,jmm
      tik(j)=t(i,j,k)*b(1,i,j,k)
10    t(i,j+1,k)=t(i,j+1,k)-b(2,i,j,k)*tik(j)

      tik(jm)=t(i,jm,k)*b(1,i,jm,k)

      do 11 j=1,jm
      t(i,j,k)=tik(j)
      t(iu,j,k)=t(iu,j,k)-b(4,i,j,k)*tik(j)
      t(il,j,ku)=t(il,j,ku)-b(6,i,j,k)*tik(j)
      t(i,j,ku)=t(i,j,ku)-b(8,i,j,k)*tik(j)
11    t(iu,j,ku)=t(iu,j,ku)-b(10,i,j,k)*tik(j)

      do 12 j=1,jmm
      t(iu,j,k)=t(iu,j,k)-b(3,i,j+1,k)*tik(j+1)
      t(i,j,ku)=t(i,j,ku)-b(7,i,j+1,k)*tik(j+1)
      t(il,j,ku)=t(il,j,ku)-b(11,i,j+1,k)*tik(j+1)
12    t(iu,j,ku)=t(iu,j,ku)-b(13,i,j+1,k)*tik(j+1)

      do 13 j=2,jm
      t(iu,j,k)=t(iu,j,k)-b(5,i,j-1,k)*tik(j-1)
      t(i,j,ku)=t(i,j,ku)-b(9,i,j-1,k)*tik(j-1)
      t(il,j,ku)=t(il,j,ku)-b(12,i,j-1,k)*tik(j-1)
13    t(iu,j,ku)=t(iu,j,ku)-b(14,i,j-1,k)*tik(j-1)
1     continue
c
100   continue
c
      return
      end
      subroutine rep(im,jm,km,t,b)
c
      dimension t(im,jm,km),b(14,im,jm,km),tik(25)
c
c
c
      kmm=km-1
      jmm=jm-1
c
c------------------remontee-------------------------------------------
c
      do 200 k=km,1,-1
c
      ku=min0(km,k+1)
c
      do 20 i=im,1,-1
c
      il=max0(1,i-1)
      iu=min0(im,i+1)
c
      do 21 j=1,jm
      tik(j)=t(i,j,k)-b(4,i,j,k)*t(iu,j,k)-b(8,i,j,k)*t(i,j,ku)-
     $b(10,i,j,k)*t(iu,j,ku)-b(6,i,j,k)*t(il,j,ku)
21    continue
      do 22 j=2,jm
      tik(j)=tik(j)-b(3,i,j,k)*t(iu,j-1,k)-b(7,i,j,k)*t(i,j-1,ku)-
     $b(11,i,j,k)*t(il,j-1,ku)-b(13,i,j,k)*t(iu,j-1,ku)
22    continue
      do 23 j=1,jmm
      tik(j)=tik(j)-b(5,i,j,k)*t(iu,j+1,k)-b(9,i,j,k)*t(i,j+1,ku)-
     $b(12,i,j,k)*t(il,j+1,ku)-b(14,i,j,k)*t(iu,j+1,ku)
23    continue
c
      t(i,jm,k)=tik(jm)*b(1,i,jm,k)

      do 30 j=jmm,1,-1
      tik(j)=tik(j)-b(2,i,j,k)*t(i,j+1,k)
30    t(i,j,k)=tik(j)*b(1,i,j,k)
c

20    continue
c
c
200   continue
c
      return
      end
