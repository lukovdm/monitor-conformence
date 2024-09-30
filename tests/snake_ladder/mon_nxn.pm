mdp

module main
    l : [0..1] init 0;

    [ladder] true -> 1:(l'=1);
    [normal] true -> 1:true;
    [snake] true -> 1:(l'=0);
endmodule

label "accepting" = l=1;
