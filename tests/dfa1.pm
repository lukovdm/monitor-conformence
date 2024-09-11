mdp

module main
    l : [0..1] init 0;

    [green] l=0 -> (l'=1);
    [red] l=0 -> (l'=0);
    [green] l=1 -> (l'=1);
    [red] l=1 -> (l'=1);
endmodule

label "accepting" = l=1;
