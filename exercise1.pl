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
% there are cards with values 
card(1,jack).
card(2,queen).
card(3,king).
card(4,ace).

% the coin has to outcomes  
heads.
tails.

%the cheeper has one in five probability of cheating 
1/5::cheater.


%%%%%%%%%%%
% Clauses %
%%%%%%%%%%%
% player1 is not affected by if player2 cheats use default probs
% There exists a draw action that can take a card of any value with 1/4 probability
1/4::draw1(X) :- card(_, X).

% if player2 is fair use default probs 
% There exists a draw action that can take a card of any value with 1/4 probability if there is no cheater
1/4::draw2(X) :- card(_, X), \+ cheater.

% if player2 cheats replaced all the jack cards with ace 
% There exists a draw action that can take a card of jack with value 1 that will never happen if there is a cheater
0::draw2(jack) :- card(1, jack), cheater.
% There exists a draw action that can take a card of ace with value 4 that will have 1/2 probability if there is a cheater
1/2::draw2(ace) :- card(4, ace), cheater.

% There exists a draw action that can take a card of any value  that is grater then 1 and less 
% then 4 with 1/2 probability if there is a cheater
1/4::draw2(X) :- card(V, X) , V>1, V<4 ,cheater.


% coin is unfair if there is a cheater
0::coin(heads); 1::coin(tails) :- cheater.

% coin is fair if nobody cheats
1/2::coin(_) :- \+ cheater.

% There exists a Same state for two card suits that have the same value and player1 and player2 have drawn thea's card.
same(D1, D2) :- card(V1, D1), draw1(D1), card(V2, D2), draw2(D2),  V1=V2.
% There exists a highest1 state for two card suits where value V1 is higher then V2 and player1 and player2 have drawn thea's card.
highest1(D1, D2) :- card(V1, D1), draw1(D1), card(V2, D2), draw2(D2),  V1>V2.
% There exists a highest2 state for two card suits where value V2 is higher then V1 and player1 and player2 have drawn thea's card.
highest2(D1, D2) :- card(V1, D1), draw1(D1), card(V2, D2), draw2(D2),  V1<V2.

% There exists a state where player 1 has the highest value but not player2 
winner(D1, D2) :- highest1(D1, D2), \+ highest2(D1, D2).

% There exists a state where player 1 and player 2 has the same card but not the coin is heads 
winner(D1, D2) :- same(D1, D2), coin(heads).

evidence(winner(jack, _), false).
evidence(winner(ace, _), true).
evidence(winner(ace, _), false).
evidence(winner(king, _), false).
evidence(winner(queen, _), false).

query(cheater).


%%%% Is Draw1 marginally independent of Coin ? ( Yes / No)    
%%%% Your answer : Yes. Since Draw1 is not a descendent of Coin, Draw1 dose not influence Coin toss and vice versa.

%%%% Is Draw1 marginally independent of Coin given Winner ? ( Yes / No)    
%%%% Your answer : No since Winner acts as a collider.

%%%% Given the observations in Table 1, learn the probability that player 2 is a cheater (keep the other parameters fixed). Use the learning tab from the online editor to do this. What is the final probability?
%%%% Your answer : Sadly dose not work... hope to get some hits.

