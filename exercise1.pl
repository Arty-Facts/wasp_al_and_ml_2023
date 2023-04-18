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
% there are two decks  
1/4::deck1(jack);
1/4::deck1(queen);
1/4::deck1(king);
1/4::deck1(ace).

1/4::deck2(jack);
1/4::deck2(queen);
1/4::deck2(king);
1/4::deck2(ace).

% cards have values
card(1,jack).
card(2,queen).
card(3,king).
card(4,ace).

%the cheeper has one in five probability of cheating 
1/5::cheater.

%%%%%%%%%%%
% Clauses %
%%%%%%%%%%%
% player1 is not affected by if player2 cheats use default probs
% There exists a draw action that can take a card of any value with 1/4 probability
draw1(X) :- deck1(X).

% if player2 is fair use default probs 
% There exists a draw action that can take a card of any value with 1/4 probability if there is no cheater
draw2(X) :- deck2(X), \+ cheater.

% if player2 cheats replaced all the jack cards with ace 
% There exists a draw action that can take a card of ace with probability of jack
draw2(ace) :- deck2(jack), cheater.

% There exists a draw action that can take a card of any value that is grater then one
draw2(X) :- deck2(X), card(V, X) , V>1, cheater.

% coin is unfair if there is a cheater
0::coin(heads); 1::coin(tails) :- cheater.

% coin is fair if nobody cheats
1/2::coin(heads); 1/2::coin(tails) :- \+ cheater.

% There exists a Same state for two card suits that have the same value and player1 and player2 have drawn thea's card.
tie(D1, D2) :- card(V1, D1), draw1(D1), card(V2, D2), draw2(D2), V1=V2.
tie :- tie(D1, D2).
% There exists a highest1 state for two draws where the card value V1 is higher 
% then V2 and player1 and player2 have drawn thea's card.
highest1(D1, D2) :- card(V1, D1), draw1(D1), card(V2, D2), draw2(D2),  V1>V2.
highest1 :- highest1(D1, D2).

% There exists a highest2 state for two draws where the card value V2 is higher 
% then V1 and player1 and player2 have drawn thea's card.
highest2(D1, D2) :- card(V1, D1), draw1(D1), card(V2, D2), draw2(D2),  V1<V2.
highest2 :- highest2(D1, D2).

% There exists a state where player 1 has the highest and its not a tie
winner1(D1, D2) :- highest1(D1, D2), \+ tie(D1, D2).
% There exists a state where player 1 and player 2 has the same card but not the coin is heads 
winner1(D1, D2) :- tie(D1, D2), coin(heads).
winner1 :- winner1(D1, D2).

% There exists a state where player 2 has the highest value but not player2 
winner2(D1, D2) :- highest2(D1, D2), \+ tie(D1, D2).
% There exists a state where player 1 and player 2 has the same card but not the coin is heads 
winner2(D1, D2) :- tie(D1, D2), coin(tails).
winner2 :- winner2(D1, D2).


%%%% Is Draw1 marginally independent of Coin ? ( Yes / No)    
%%%% Your answer : Yes. Since Draw1 is not a descendent of Coin, Draw1 dose not influence Coin toss and vice versa.

%%%% Is Draw1 marginally independent of Coin given Winner ? ( Yes / No)    
%%%% Your answer : No since Winner acts as a collider. An thus affect the outcome.

%%%% Given the observations in Table 1, learn the probability that player 2 is a cheater (keep the other parameters fixed). Use the learning tab from the online editor to do this. What is the final probability?
%%%% Your answer : 

% evidence(draw1(jack)).
% evidence(winner2).
% ---
% evidence(draw1(ace)).
% evidence(winner1).
% ---
% evidence(draw1(ace)).
% evidence(winner2).
% ---
% evidence(draw1(king)).
% evidence(winner1).
% ---
% evidence(draw1(queen)).
% evidence(winner2).

% Answer: t(_)::cheater = 0.6506477
