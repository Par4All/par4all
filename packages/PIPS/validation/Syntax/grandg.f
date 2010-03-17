com ===================================================================
      subroutine grandg(ifsim)
com ===================================================================
      implicit double precision (a-h,o-z)
C ##### include: siddimension.h #####
com =======Header siddimension.h===============================                 
com Definition des dimensions du probleme                                       
c -------------------------------------------------------------------+          
      parameter (maxcoef=  20)
      parameter (maxcara=  2)
      parameter (maxvobs=  5)
      parameter (maxvint=  28100)
      parameter (maxvaux=  4)
c -------------------------------------------------------------------+          
      parameter (maxtest=  10)
      parameter (maxseq =  200)
      parameter (maxchar=  7)
      parameter (maxpexp=  100)
      parameter (maxpsim=  100)
      parameter (maxccha=  4)
      parameter (maxwind=  3)
c -------------------------------------------------------------------+                
com   maxcoef = nombre de coefficients dans cet executable                            
com   maxcara = nombre de caracteristiques experimentales                             
com                          (option caracteristique de *essai)                       
com   maxvobs= nombre de variables "observees"                                        
com   maxvint= nombre de variables a integrer                                         
com   maxvaux= nombre de variables auxiliaires a evaluer en sortie                    
com   maxtest = nombre max d essais possibles pour cet executable                     
com   Dans le cas ou on definit le chargement dans un fichier :                       
com  -maxseq  = nombre de max de sequences (segments de droite pour                   
com             definir le chargement)                                                
com  -maxchar= nombre de colonnes du fichier chargement                               
com   maxpexp= nombre  maxi d observations experimentales                             
com   maxpsim= nombre de points a retenir pour la simulation                          
com   maxccha= nombre max de variables pour la description du chargement              
com            par une formule                                                        
com   maxwind= nombre max de fenetres (intervalles d integration) pour                
com            lesquelles on peut effectuer des sorties pour un essai                 
c ============End of siddimension.h======================================               
C ##### end include #####
C ##### include: sidvar.h #####
com =======Common sidvar.h==============================================
com -------------------------------------------------------------------+
      common/cocoef/
     .  young,poiss,grcmac,grdmac,
     .  grcgra,grdgra,
     .  xn,xk,rayvi,granq,petib,petic0,petic1,petid,
     .  hcfc1,hcfc2,hcfc3,hcfc4,hcfc5,hcfc6
      common/cocara/
     .  diamext,diamint
      common/varobs/
     .  temps,force,couple,eto11,eto12,nul1,nul2
      common/varaux/
     .  sigmi,sigtr,evimi,evitr
      common/varint/
     .  sig11,sig22,sig33,sig12,sig23,sig13,
     .  evi11,evi22,evi33,evi12,evi23,evi13,
     .  bet11,bet22,bet33,bet12,bet23,bet13,
     .  evcum,gamet,
     .  betgr(5616),
     .  gvcum(11232),almic(11232)
      common/dvarint/
     .  dsig11,dsig22,dsig33,dsig12,dsig23,dsig13,
     .  devi11,devi22,devi33,devi12,devi23,devi13,
     .  dbet11,dbet22,dbet33,dbet12,dbet23,dbet13,
     .  devcum,dgamet,
     .  dbetgr(5616),
     .  dgvcum(11232),dalmic(11232)
c   =====End of sidvar.h===============================================
C ##### end include #####
C ##### include: sidcristal.h #####
com =======Header sidcristal.h===============================  

c on stocke les tenseurs d orientation de 
c tous les grains*tous les systemes >> ngrain*nbsys*ntens valeurs
c
      common /icristal/nbsys,ngrain,ntotdir
      common /rcristal/hcfc(12,12),xm(6,12,1000)

c  ==========End of sidcristal.h==============================
C ##### end include #####
C ##### include: sidcalc.h #####
com ==Common sidcalc.h===================================================

com Common contenant quelques petites variables tres critiques
com -------------------------------------------------------------------+
      common/icalcul/i1fois,itest,npt,nctrl,jchold,numite,maxfun,
     . ifpauto,ifsimul,ifverif,ifformu(maxtest),iffenetre(maxtest),
     . ntest,nscal,nvect,nscal2,nvect2
      common/rcalcul/char2(maxccha,maxvobs),hh,dmax,acc
      common/ccalcul/varko(maxchar)
      character*8 varko
c   -------------------------------------------------------------------+
com A completer
c   =====End of sidcalc.h=================================================
C ##### end include #####
C ##### include: sidinout.h #####
com =======Common sidinout.h============================================
com Common des entrees-sorties
com -------------------------------------------------------------------+
      common/choisor/nsscal,nsvect,lrsscal(maxvint),lrsvect(maxvint),
     .              lsvect(100)
      common/inout/iecran,iclavi,lonrec,iuti,idat,isca,ivec,iaux,
     . iexp,ilst,itrj,itra,icof,iprint,ifsorauto,lonfich
      common/cinout/fichnom,fichexp,fichs,fichtest,fichtrj,fichcof,
     .              titre,auteur,repere
      character*16 fichs,fichnom,fichexp(maxtest),fichtest(maxtest),
     .             fichtrj,fichcof
      character*80 titre,auteur,repere
c   -------------------------------------------------------------------+
com A completer 
c   ========End of sidinout.h============================================
C ##### end include #####
      character*16 filsys
c=========================================================
      data ze/0.0d0/ un/1.0d0/ de/2.0d0/ tr/3.0d0/ qu/4.0d0/
      data pi4/0.785398163d0/ ct/1.0d0/
      de2=diamext*diamext
      di2=diamint*diamint
      air=pi4*(de2-di2)
      bir=pi4*(de2*diamext-di2*diamint)/tr
      force=air*sig11
      if (temps.lt.1200.0) then
        couple=bir*sig12
      else
        couple=ze
      end if
      eto11=(sig11/young+evi11)*ct
      dmoy=(diamint+diamext)/de
      eto12=(sig12*(un+poiss)/young+evi12)*ct*diamext/dmoy
      if (ifsim.ne.1) return
      sigmi=dsqrt(sig11*sig11+tr*sig12*sig12)
      sigtr=dsqrt(sig11*sig11+qu*sig12*sig12)
      evimi=dsqrt(evi11*evi11+qu*evi12*evi12/tr)
      evitr=dsqrt(evi11*evi11+qu*qu*evi12*evi12/tr/tr)
      write(6,'(5(2x,e13.5))') temps,sig11,sig12,eto11,eto12
      if (npt.ge.maxpsim-1) then
        lonfich=index(fichtest(itest),' ')-1
        filsys=fichtest(itest)(1:lonfich)//'.sys'
C*        filsys=fichtest(itest)(1:lonfich)
        open(unit=56,file=filsys)
        do 910 igrain=1,ngrain
         isys=12*(igrain-1)
         write(56,*) 'grain ',igrain
         write(56,'(6(2x,f9.5))') (gvcum(id),id=isys+1,isys+6)
         write(56,'(6(2x,f9.5))') (gvcum(id),id=isys+7,isys+12)
  910   continue
        close(56)
      end if
      end
