%%%% The Problog program implementing the Bayesian network.
% Cheater -> Draw1, Cheater -> Draw2, Cheater -> Coin,
% Draw1 -> Highest
% Draw2 -> Highest
% Highest -> Coin
% Highest -> Winner
% Coin -> Winner

%%%%%%%%%
% Facts %
%%%%%%%%%
heads.
tails.

%%%%%%%%%%%%%%%%%%%%%%%
% Probabilistic Facts %
%%%%%%%%%%%%%%%%%%%%%%%
1/5::cheater.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Probabilistic Fact Disjunctions %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% default probs
1/4::jack;
1/4::queen; 
1/4::king;
1/4::ace.

%%%%%%%%%%%
% Clauses %
%%%%%%%%%%%
% player1 is not affected by if player2 cheats use default probs
draw1(X) :- X.
% if player2 is fair use default probs  
draw2(X) :- X, \+ cheater.

%%%%%%%%%%%%%%%%%%%%%%%%%
% Probabilistic Clauses %
%%%%%%%%%%%%%%%%%%%%%%%%%
% if player2 cheats replaced all the jack cards with ace 
0::draw2(jack) :- cheater.
1/2::draw2(ace) :- cheater.

% coin is fair if nobody cheats
1/2::coin(_) :- \+ cheater.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Probabilistic Clause Disjunctions %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% coin is unfair if there is a cheater
0::coin(heads); 1::coin(tails) :- cheater.

%%%%%%%%%%
% Querys %
%%%%%%%%%%

evidence(cheater, true).

query(draw1(jack)).
query(draw2(jack)).
query(draw2(ace)).
query(coin(heads)).


%%%% Is Draw1 marginally independent of Coin ? ( Yes / No)    
%%%% Your answer : 

%%%% Is Draw1 marginally independent of Coin given Winner ? ( Yes / No)    
%%%% Your answer : 

%%%% Given the observations in Table 1, learn the probability that player 2 is a cheater (keep the other parameters fixed). Use the learning tab from the online editor to do this. What is the final probability?
%%%% Your answer : 



% TEST
%%%%%%%%%
% Facts %
%%%%%%%%%
heads.
tails.
card(1,jack). 
card(2,queen). 
card(3,king).
card(4,ace).

%%%%%%%%%%%%%%%%%%%%%%%
% Probabilistic Facts %
%%%%%%%%%%%%%%%%%%%%%%%
1/5::cheater.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Probabilistic Fact Disjunctions %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% default probs
1/4::jack;
1/4::queen; 
1/4::king;
1/4::ace.

%%%%%%%%%%%
% Clauses %
%%%%%%%%%%%
% player1 is not affected by if player2 cheats use default probs
draw1(X) :- X.
% if player2 is fair use default probs  
draw2(X) :- X, \+ cheater.

%%%%%%%%%%%%%%%%%%%%%%%%%
% Probabilistic Clauses %
%%%%%%%%%%%%%%%%%%%%%%%%%
% if player2 cheats replaced all the jack cards with ace 
0::draw2(jack) :- cheater.
1/2::draw2(ace) :- cheater.

% coin is fair if nobody cheats
1/2::coin(_) :- \+ cheater.
values(v1, v2) :- card(v1, d1), card(v2, d2).
%highest(h) :- 
%    draw1(d1), 
%    draw2(d2), 
%    card(v1, d1), 
%    card(v2, d2),
%    v1 < v2, 
%    h is d2.
    
%highest(h) :- draw1(d1), draw2(d2), card(v1, d1), card(v2, d2), v1 > v2, h is d2.

%highest(h) :- draw1(d1), draw2(d2), card(v1, d1), card(v2, d2), v1 = v2, coin(tails), h is d2.
%highest(h) :- draw1(d1), draw2(d2), card(v1, d1), card(v2, d2), v1 = v2, coin(heads), h is d1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Probabilistic Clause Disjunctions %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% coin is unfair if there is a cheater
0::coin(heads); 1::coin(tails) :- cheater.

%%%%%%%%%%
% Querys %
%%%%%%%%%%

evidence(cheater, true).

%query(draw1(jack)).
%query(draw2(jack)).
%query(draw2(ace)).
%query(coin(heads)).
query(values(_, _)).