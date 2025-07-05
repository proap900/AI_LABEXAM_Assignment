/*
Develop rules for brother-in-law and sister-in-law and experiment with different forms of queries and explain.
*/
/* Facts */
man(ram).
man(laxuman).
man(shyam).
woman(sita).
woman(urmila).

father(janak, sita).
father(janak, urmila).
father(janak, shyam).
father(dasarath, ram).
father(dasarath, laxuman).

/* Rules */

sibling(X, Y) :-
    father(Z, X),
    father(Z, Y),
    X \= Y.

% Marriage Validation
valid_marriage(X, Y) :-
    man(X), woman(Y),
    \+ sibling(X, Y),
    !.

valid_marriage(X, Y) :-
    woman(X), man(Y),
    \+ sibling(X, Y),
    !.

% Marriage facts with constraint check
% Use dynamic declaration so we can add facts at runtime
:- dynamic marry/2.

% Custom predicate to add marriage
marry(X, Y) :-
    valid_marriage(X, Y),
    assertz(marriage_fact(X, Y)),
    !.

% Define a helper to check marriage in either direction
marriage_fact(ram, sita).
marriage_fact(laxuman, urmila).

spouse(X, Y) :- marriage_fact(X, Y).
spouse(X, Y) :- marriage_fact(Y, X).

/* Brother-in-law: 
1. Husband of one's sibling
2. Brother of one's spouse
*/
brother_in_law(X, Y) :-
    man(X),
    sibling(Y, S),
    spouse(X, S).

brother_in_law(X, Y) :-
    man(X),
    spouse(Y, S),
    sibling(X, S).

/* Sister-in-law:
1. Wife of one's sibling
2. Sister of one's spouse
*/
sister_in_law(X, Y) :-
    woman(X),
    sibling(Y, S),
    spouse(X, S).

sister_in_law(X, Y) :-
    woman(X),
    spouse(Y, S),
    sibling(X, S).
