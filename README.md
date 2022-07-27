# Stage_Master_2
 
Ce repository contient les différents codes ainsi que plusieurs simulations, tous réalisés pendant un stage de deuxième année de Master au sein de l'équipe MIMESIS et de l'équipe MLMS. L'un des objectifs de ce stage était d'implémenter un schéma <img src="https://render.githubusercontent.com/render/math?math=\phi">-FEM pour les problèmes hyperélastiques. Pour cela, dans le dossier `./Codes_python/simulations_fenics` se trouvent deux dossiers : `2D` et `3D`. 
Chaque dossier contient les codes permettant d'appliquer la méthode <img src="https://render.githubusercontent.com/render/math?math=\phi">-FEM et la méthode standard pour les problèmes

* de Poisson avec conditions de Dirichlet pures,
* de Poisson avec conditions de Neumann pures,
* de Poisson avec conditions mixtes,
* de Poisson non-linéaire avec des conditions mixtes,
* d'élasticité linéaire, 
* d'hyperélasticité. 

Dans le cas de problèmes 2D, chaque problème peut être appliqué à un cercle ou à une ellipse, et pour les problèmes 3D, à une sphère ou une ellipsoïde.
 
Dans le dossier `./Codes_python` se trouve également un fichier rassemblant les codes utilisés pour réaliser les représentations graphiques faites dans le rapport. 

Le dossier `./Codes_sofa` contient lui deux simulations réalisées avec Sofa et SofaCaribou, ainsi qu'un dossier `codes_caribou_fictitious_grid` dans lequel se trouvent deux codes permettant de tester les résultats obtenus à la suite des modification faites de le code source de Caribou.

Enfin, dans le dossier `./videos_simulations_sofa` se trouvent les résultats vidéos des deux simulations obtenues avec les codes `./Codes_sofa/elastic_ball_bouncing.py` et `./Codes_sofa/liver_beam.py`
