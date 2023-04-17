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
1/4::suit(jack);
1/4::suit(queen);
1/4::suit(king);
1/4::suit(ace).

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
draw1(X) :- suit(X).

% if player2 is fair use default probs 
% There exists a draw action that can take a card of any value with 1/4 probability if there is no cheater
draw2(X) :- suit(X), \+ cheater.

% if player2 cheats replaced all the jack cards with ace 
% There exists a draw action that can take a card of ace with probability of jack
draw2(ace) :- suit(jack), cheater.

% There exists a draw action that can take a card of any value that is grater then one
draw2(X) :- suit(X), card(V, X) , V>1, cheater.

% coin is unfair if there is a cheater
0::coin(heads); 1::coin(tails) :- cheater.

% coin is fair if nobody cheats
1/2::coin(_) :- \+ cheater.

% There exists a Same state for two card suits that have the same value and player1 and player2 have drawn thea's card.
same(draw1, draw2) :- draw1(D), draw2(D).

% There exists a highest1 state for two draws where the card value V1 is higher 
% then V2 and player1 and player2 have drawn thea's card.
highest1(draw1, draw2) :- card(V1, D1), draw1(D1), card(V2, D2), draw2(D2),  V1>V2.

% There exists a highest2 state for two draws where the card value V2 is higher 
% then V1 and player1 and player2 have drawn thea's card.
highest2(draw1, draw2) :- card(V1, D1), draw1(D1), card(V2, D2), draw2(D2),  V1<V2.

% There exists a state where player 1 has the highest value but not player2 
winner(D1, D2) :- highest1(D1, D2), \+ highest2(D1, D2).

% There exists a state where player 1 and player 2 has the same card but not the coin is heads 
winner(D1, D2) :- same(D1, D2), coin(heads).

evidence(cheater, false).
%evidence(draw1(king), true).

query(highest1(_,_)).
 

%query(draw2(_)).


%%%% Is Draw1 marginally independent of Coin ? ( Yes / No)    
%%%% Your answer : Yes. Since Draw1 is not a descendent of Coin, Draw1 dose not influence Coin toss and vice versa.

%%%% Is Draw1 marginally independent of Coin given Winner ? ( Yes / No)    
%%%% Your answer : No since Winner acts as a collider.

%%%% Given the observations in Table 1, learn the probability that player 2 is a cheater (keep the other parameters fixed). Use the learning tab from the online editor to do this. What is the final probability?
%%%% Your answer : Sadly dose not work... hope to get some hits.

