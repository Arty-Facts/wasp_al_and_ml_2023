%%%% The Problog program implementing the Bayesian network.
% Cheater -> Draw2, Cheater -> Coin,
% Draw1 -> Highest
% Draw2 -> Highest
% Highest -> Coin
% Highest -> Winner
% Coin -> Winner

%%%%%%%%%
% Facts %
%%%%%%%%%
% default probs
card(1,jack).
card(2,queen).
card(3,king).
card(4,ace).

heads.
tails.

1/5::cheater.


%%%%%%%%%%%
% Clauses %
%%%%%%%%%%%
% player1 is not affected by if player2 cheats use default probs
1/4::draw1(X) :- card(_, X).

% if player2 is fair use default probs  
1/4::draw2(X) :- card(_, X), \+ cheater.
% if player2 cheats replaced all the jack cards with ace 
0::draw2(jack) :- card(1, jack), cheater.
1/2::draw2(ace) :- card(4, ace), cheater.
1/4::draw2(X) :- card(V, X) , V>1, V<4 ,cheater.


% coin is unfair if there is a cheater
0::coin(heads); 1::coin(tails) :- cheater.

% coin is fair if nobody cheats
1/2::coin(_) :- \+ cheater.

same(D1, D2) :- card(V1, D1), draw1(D1), card(V2, D2), draw2(D2),  V1=V2.
highest1(D1, D2) :- card(V1, D1), draw1(D1), card(V2, D2), draw2(D2),  V1>V2.
highest2(D1, D2) :- card(V1, D1), draw1(D1), card(V2, D2), draw2(D2),  V1<V2.

winner(D1, D2) :- highest1(D1, D2), \+ highest2(D1, D2).
winner(D1, D2) :- same(D1, D2), coin(heads).

winner :- highest1(D1, D2), \+ highest2(D1, D2).
winner :- same(D1, D2), coin(heads).

evidence(cheater, false).
%evidence(draw1(jack)).
%evidence(draw2(king)).

%query(highest1(_, _)).
%query(highest2(_, _)).
%query(same(_, _)).
%query(coin(heads)).
query(winner).


%%%% Is Draw1 marginally independent of Coin ? ( Yes / No)    
%%%% Your answer : Yes. Since Draw1 is not a descendent of Coin, Draw1 dose not influence Coin toss and vice versa.

%%%% Is Draw1 marginally independent of Coin given Winner ? ( Yes / No)    
%%%% Your answer : No since Winner acts as a collider.

%%%% Given the observations in Table 1, learn the probability that player 2 is a cheater (keep the other parameters fixed). Use the learning tab from the online editor to do this. What is the final probability?
%%%% Your answer : 

