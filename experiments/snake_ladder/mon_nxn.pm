mdp

module main
    l : [0..1] init 0;

    [normal] true -> 1:(l'=1);
    [ladder] true -> 1:(l'=1);
    [snake] true -> 1:(l'=1);
endmodule

label "accepting" = l=1;
