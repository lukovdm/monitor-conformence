mdp

const horizon = 20;
formula horizon_step = min(step+1, horizon);

module main
    l : [0..1] init 0;

    [normal] step>=0 -> 1:(l'=1);
    [ladder] step>=0 -> 1:(l'=1);
    [snake] step>=0 -> 1:(l'=1);
    // [ladder] l=0 & step=0 -> 1:(l'=0);
    // [normal] l=0 & step=1 -> 1:(l'=0);
    // [ladder] l=0 & step=2 -> 1:(l'=0);
    // [normal] l=0 & step=3 -> 1:(l'=0);
    // // [snake] l=0 & step=4 -> 1:(l'=0);
    // [normal] step>3 -> 1:(l'=1);
    // [ladder] step>3 -> 1:(l'=1);
    // [snake] step>3 -> 1:(l'=1);

    // [snake] l=0 & step>1 -> 1:(l'=1);
    // [ladder] l=0 & step>1 -> 1:(l'=0);
    // [normal] l=0 & step>1 -> 1:(l'=0);
    // [normal] l=1 & step>1 -> 1:(l'=1);
    // [ladder] l=1 & step>1 -> 1:(l'=1);
    // [snake] l=1 & step>1 -> 1:(l'=1);
endmodule

module hor
    step : [0..horizon] init 0;
    [ladder] true -> 1:(step'=horizon_step);
    [normal] true -> 1:(step'=horizon_step);
    [snake] true -> 1:(step'=horizon_step);
endmodule

label "horizon" = step=horizon;

label "accepting" = l=1;
