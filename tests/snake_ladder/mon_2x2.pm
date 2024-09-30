mdp

module main
    l : [0..2] init 0;

    [normal] (l=0) -> 1:(l'=1);
    [ladder] (l=1) -> 1:(l'=2);
    [normal] (l=2) -> 1:(l'=2);
endmodule

label "accepting" = l=2;