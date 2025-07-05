% ============================================
% LAB EXAM: PROLOG Programming
% 1. Family Tree
% 2. Graph Routing (Directed & Undirected)
% 3. Pattern/Numerical Series
% ============================================

% --------------------------------------------
% 1. FAMILY TREE RELATIONSHIPS
% --------------------------------------------

% Gender facts
male(john).
male(paul).
male(david).
female(mary).
female(lisa).
female(susan).

% Parent-child relationships
parent(john, paul).
parent(john, mary).
parent(paul, david).
parent(mary, lisa).
parent(mary, susan).

% Rules
father(X, Y) :- parent(X, Y), male(X).
mother(X, Y) :- parent(X, Y), female(X).
sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \= Y.
grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
grandfather(X, Y) :- grandparent(X, Y), male(X).
grandmother(X, Y) :- grandparent(X, Y), female(X).
uncle(X, Y) :- male(X), sibling(X, Z), parent(Z, Y).
aunt(X, Y) :- female(X), sibling(X, Z), parent(Z, Y).

% --------------------------------------------
% 2. GRAPH ROUTING
% --------------------------------------------

% ---------- Directed Graph ----------
% edge(From, To)
edge_directed(a, b).
edge_directed(b, c).
edge_directed(c, d).
edge_directed(a, e).
edge_directed(e, d).

% Path in directed graph
path_directed(X, Y) :- edge_directed(X, Y).
path_directed(X, Y) :- edge_directed(X, Z), path_directed(Z, Y).

% ---------- Undirected Graph ----------
% connected(X, Y) is bidirectional
connected(a, b).
connected(b, c).
connected(c, d).
connected(b, e).
connected(e, d).

% Convert to bidirectional edge
edge_undirected(X, Y) :- connected(X, Y).
edge_undirected(Y, X) :- connected(X, Y).

% Wrapper predicate
path_undirected(X, Y) :- path_undirected(X, Y, [X]).

% Base case: reached destination
path_undirected(X, X, _).

% Recursive case with cycle check
path_undirected(X, Y, Visited) :-
    edge_undirected(X, Z),
    \+ member(Z, Visited),
    path_undirected(Z, Y, [Z | Visited]).

% --------------------------------------------
% 3. NUMERICAL / PATTERN-BASED SERIES
% --------------------------------------------

% ---------- Fibonacci Series ----------
fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.



% ----------fibonacci Series (upto nth term) ----------
fib_series(N, Series) :-
    fib_series_helper(0, N, Series).

fib_series_helper(N, N, []) :- !.
fib_series_helper(I, N, [F|Rest]) :-
    I < N,
    fib(I, F),
    I1 is I + 1,
    fib_series_helper(I1, N, Rest).


% ---------- Natural Numbers (Generate Infinite) ----------
natural(1).
natural(N) :- natural(M), N is M + 1.

% ---------- Power of 2 ----------
power_of_2(0, 1).
power_of_2(N, P) :-
    N > 0,
    N1 is N - 1,
    power_of_2(N1, P1),
    P is P1 * 2.