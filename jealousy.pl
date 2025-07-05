/*
Develop rules for the following: Jealousy between two persons if they love same person of opposite gender. Experiment with different forms of queries and explain.*/
man(ram).
man(ravan).
woman(sita).
woman(urmila).
loves(ram,sita).
loves(ravan,sita).
loves(ravan,urmila).
jealousy(X,Z):- loves(X,Y),loves(Z,Y).
jealousy(Y,Z):-loves(X,Y),loves(X,Z).