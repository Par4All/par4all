Transformation de programmes multi-cibles
=========================================

*Multi-target program transformations*

Sujet de stage d’ingénieur ou M2

**Mots-clés** : compilation, parallélisation, accélérateurs matériels,
transformation de programmes, optimisations

Afin de fournir aux utilisateurs toujours plus de puissance de calcul, on
assiste actuellement au déploiement de nombreuses architectures parallèles
bon marché basées sur différentes technologies : processeurs multi-cœurs,
multiprocesseurs hétérogènes sur puce (Cell, Larrabee…), des coprocesseurs
graphiques (GPGPU nVidia, AMD/ATI…) ou des accélérateurs à base de
circuits logiques reconfigurables (FPGA).

Malheureusement on assiste à une profusion des modèles de programmation et
des outils disponibles pour chaque cible. Afin de faciliter la
programmation de ses architectures prometteuses, on se propose d’utiliser
les techniques de parallélisation automatique et de transformation de
programmes pour générer du code de manière portable pour différentes
cibles à partir de programmes séquentiels classiques écrits en C ou
Fortran, éventuellement décorés de directives (à la OpenMP) par le
programmeur.

Le travail consistera à réaliser des phases de transformation (fusion de
boucles, pavage de boucles...) et à factoriser des phases déjà existantes
dans l’outil de transformation libre RoseCompiler
(http://www.rosecompiler.org).

Profil recherché : passionné d’informatique voulant faire de la vraie
informatique et n’ayant pas peur de réfléchir (formalisme mathématique et
sémantique) ni de programmer. Bonnes connaissances utiles si possible :
C++, C, Python, Unix, programmation parallèle..

Ce stage peut enchaîner avec une embauche et/ou une thèse.

Encadrant : Ronan Keryell (rk at hpc-project dot com) http://par4all.org

Lieu : HPC Project http://www.hpc-project.com Une start-up de 35 personnes
en pleine croissance.

Meudon (92) ou Montpellier (34), France.

..
  # Some Emacs stuff:
  ### Local Variables:
  ### mode: rst,flyspell
  ### ispell-local-dictionary: "french"
  ### End:
