:-use_module(library(apply)).
:-use_module(library(lists)).

%%%% Insert and modify the ProbLog code from Exercise 1 here

% Encode the different cards as follows: card(Player,N,Rank)
% This means that the N-th card drawn by Player is of the given Rank.

% this is the order for card ranks from the weakest to the strongest hand
ranks([jack, queen, king, ace]).

cards_in_hand([1, 2, 3, 4]).

players([player1, player2]).
% this is the order for hand ranks from the weakest to the strongest hand
hand_ranks([highcard, onepair, twopair, threeofakind, straight, fullhouse, fourofakind]).

% the cheeper has one in five probability of cheating 
1/5::cheater(T_ID):- % need T_ID for trial identifier
    card_nb(T_ID).

% coin is unfair if there is a cheater
0::coin(heads); 1::coin(tails) :- cheater(T_ID).

% coin is fair if nobody cheats
1/2::coin(heads); 1/2::coin(tails) :- \+ cheater(T_ID).
% evidence(cheater(T_ID)).
% query(coin(_)).

%% Define the cards
card(Player, N, Rank) :-
    player(Player),
    draw(Player, N, Rank).
    % write(Player), write(' was dealt '), write(N), write(Rank), nl.

player(Player) :-
    players(PlayerList),
    member(Player, PlayerList).

1/NbCards::draw(player1, N, Rank):- 
    ranks(RankList),
    member(Rank, RankList), 
    length(RankList, NbCards),
    card_nb(N).

1/NbCards::draw(player2, N, Rank):-
    valid_card(Rank),
    ranks(RankList),
    length(RankList, NbCards), 
    card_nb(N), 
    \+ cheater(N).

2/NbCards::draw(player2, N, ace):-
    ranks(RankList),
    length(RankList, NbCards),
    card_nb(N),
    cheater(N).

1/NbCards::draw(player2, N, Rank) :-
    valid_card(Rank),
    ranks(RankList),
    Rank \= jack, % jack and ace are accounted for in statement above  
    Rank \= ace, 
    length(RankList, NbCards),
    card_nb(N),
    cheater(N).

card_nb(N):-
    cards_in_hand(Nth), 
    member(N, Nth).

card_value(Value, Rank):-
    valid_card(Rank),
    ranks(RankList),
    nth0(Value, RankList, Rank).

valid_card(Rank):-
    ranks(RankList),
    member(Rank, RankList).

% evidence(cheater(N)).
% evidence(card(player2, 1, jack)).
% query(card(player1, _, _)).
% query(card(player2, 1, _)).

%% Define the poker hands
     
% A straight is a hand that contains five cards of sequential rank.
% but according to the test its only four
% hand(Cards, straight(R1, R2, R3, R4)) :-
hand(Cards, straight) :-
    select(R1, Cards, Rest1), card_value(V1, R1),
    select(R2, Rest1, Rest2), card_value(V2, R2),
    V1 + 1 =:= V2, %early terminate if possible to improve performance
    select(R3, Rest2, Rest3), card_value(V3, R3),
    V2 + 1 =:= V3,
    member(R4, Rest3), card_value(V4, R4),
    V3 + 1 =:= V4.

% A full house is a hand that contains three cards of one rank and two cards of 
% another rank    
hand(Cards, fullhouse(Rank1, Rank2)) :-
    hand(Cards, threeofakind(Rank1)), 
    hand(Cards, onepair(Rank2)), 
    Rank1 \= Rank2, 
    valid_card(Rank1),valid_card(Ran2).

hand(Cards, fourofakind(Rank)) :-
    valid_card(Rank),
    select(Rank, Cards, Rest1),
    select(Rank, Rest1, Rest2),
    select(Rank, Rest2, Rest3),
    member(Rank, Rest3).

hand(Cards, threeofakind(Rank)) :-
    valid_card(Rank),
    select(Rank, Cards, Rest1),
    select(Rank, Rest1, Rest2),
    member(Rank, Rest2).

hand(Cards, twopair(Rank1, Rank2)) :-
    hand(Cards, onepair(Rank1)),
    hand(Cards, onepair(Rank2)),
    Rank1 \= Rank2, 
    valid_card(Rank1),valid_card(Ran2).
    
hand(Cards, onepair(Rank)) :-
    valid_card(Rank),
    select(Rank, Cards, Rest1),
    member(Rank, Rest1).
    
hand(Cards, highcard(Rank)) :-
    valid_card(Rank),
    member(Rank, Cards).

%query(hand([poop, poop, jack, king, jack, jack, queen, ace, jack,jack, ace, queen, queen, king, ace], _)).

better(Hand1, Hand2):-
    hand_value(Hand1, Value1),
    hand_value(Hand2, Value2),
    Value1 > Value2.
    
better(Hand1, Hand2):-
    hand_value(Hand1, Value1),
    hand_value(Hand2, Value2),
    Value1 =:= Value2, 
    decompose(Hand1, _, Ranks1),
    decompose(Hand2, _,  Ranks2),
    compare_in_rank(Ranks1, Ranks2).

hand_value(Hand, Value):-
    hand_ranks(HandRankList),
    decompose(Hand, HandRank, _),
    member(HandRank, HandRankList),
    nth0(Value, HandRankList, HandRank).

compare_in_rank([Rank1| _], [Rank2| _]) :-
       card_value(Value1, Rank1), 
       card_value(Value2, Rank2),
       Value1 > Value2.

compare_in_rank([Rank1| Tail1], [Rank2| Tail2]) :-
       card_value(Value1, Rank1), 
       card_value(Value2, Rank2),
       Value1 =:= Value2,
       compare_in_rank(Tail1, Tail2).

decompose(Hand, HandRank, Ranks) :-  
       Hand =.. [HandRank| Ranks]. % decompose ex fullhouse([king, jack]) to [fullhouse, [king, jack]]

% query(decompose(fullhouse(king,queen),_, _)).
% query(compare_in_rank([king,queen],[king,jack])).
% query(better(fullhouse(king,queen),fullhouse(king,jack))).
% query(best_hand([jack, king, jack, ace],_)).


%%%% Provided code

% The following predicate will sample a Hand as a list of ranks for the given player.
% It expects that there are probabilistic facts of the form card(Player,N,Rank) as specified above

draw_hand(Player,Cards) :- cards_in_hand(CardIndex), maplist(card(Player), CardIndex ,Cards).

% evidence(cheater(N)).
% query(draw_hand(player1,_)).
% query(draw_hand(player2,_)).

game_outcome(Cards1,Cards2,Outcome) :-
    best_hand(Cards1,Hand1),
    best_hand(Cards2,Hand2),
    outcome(Hand1,Hand2,Outcome).

outcome(Hand1,Hand2,player1) :- better(Hand1,Hand2).
outcome(Hand1,Hand2,player2) :- better(Hand2,Hand1).
outcome(Hand1,Hand2,player1) :- \+better(Hand1,Hand2), \+better(Hand2,Hand1), coin(heads).
outcome(Hand1,Hand2,player2) :- \+better(Hand1,Hand2), \+better(Hand2,Hand1), coin(tails).

best_hand(Cards,Hand) :-
    hand(Cards,Hand),
    \+ (hand(Cards,Hand2), better(Hand2,Hand)).

winner(Outcome):- 
    player(Outcome), 
    draw_hand(player1,Cards1), 
    draw_hand(player2,Cards2), 
    game_outcome(Cards1,Cards2,Outcome). 
%winner(Outcome):- draw_hand(player1,Hand1), draw_hand(player2,Hand2), player(Outcome), outcome(Hand1,Hand2,Outcome).
% winner(Outcome) :- winner(Outcome, Hand1, Hand2).
%%%% What’s the probability that player2 draws the hand [ace, king, queen, ace].
%%%% Your answer : query(draw_hand(player2,[ace, king, queen, ace])).:  0.005625


%%%%  Given that player2 draws the hand [ace, king, queen, ace], and that the coin comes up tails, 
%%%%  what is the posterior belief that your opponent is cheating?
%%%% Your answer : t(_)::cheater(T_ID): 1	
% ---
% evidence(coin(tails)).
% evidence(draw_hand(player2,[ace, king, queen, ace])).
% ---


%%%%  What is the prior probability that player 1 wins?
%%%% 1 Why does this query take so long to answer? What is the probability that player 1 wins, given that you know that player 2 is a cheater?
%%%% Your answer : 
% Q1
% query(winner(player1)).
% winner(player1):        0.1562878
% winner(player2):        0.1834804
% Well player1 has 4 cards and there are 4^4 permutation of hands and every hand can hold many probabilities for the most likely best hands.
% The same hold for player2 so one can se how the number computation explodes with O(n^4) and since every branch need to be 
% evaluated to compute the probability.
% 
% Q2
% evidence(cheater(N)).
% query(winner(player1)).
% winner(player1):        0.11541525
% winner(player2):        0.22948785
% 
% RANT: Playing with 2 card in hand is more feasible and the results presented are from that 
% commutation. One thing that frustrates me is that the probabilities for player1 and player2 does not add up to 1 but i cant 
% solve since i have spent way to match time on this lab hence i as for help from you!
% This was a fun lab to do but i did not expect spending 60h to explore the intreating landscape of Prolog and Problog.  
% If this lab will be offered for other student a in the future a better cheat sheet is needed since documentation is lacking 
% and all Prolog features are not supported. But nice to knows are how to decompose statements how to identify issues with stochastic 
% memoization and how to use trial identifier. Is there a debugger and or a profiler?  
% My final thoughts im very open for feedback since i can se myself using Problog in the future but im not 
% really sure about best practices. Since its first time for me using Prolog and Problog I assume the there are many weird stuff
% in the code and i would love to get feedback how to improve.
% thanks.      