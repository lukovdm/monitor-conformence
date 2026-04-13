mdp

module main
    l : [0..3] init 0;

    [normal] (l=0) -> 1:(l'=1);
    [ladder] (l=1) -> 1:(l'=2);
    [normal] (l=2) -> 1:(l'=2);
    [b] true -> 1:(l'=3);
endmodule

label "accepting" = l=2;