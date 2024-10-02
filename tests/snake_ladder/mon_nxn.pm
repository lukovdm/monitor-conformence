mdp

const horizon = 10;
formula horizon_step = min(step+1, horizon);

module main
    l : [0..1] init 0;

    [ladder] true -> 1:(l'=0);
    [normal] true -> 1:true;
    [snake] true -> 1:(l'=1);
endmodule

module hor
    step : [0..horizon] init 0;
    [ladder] true -> 1:(step'=horizon_step);
    [normal] true -> 1:(step'=horizon_step);
    [snake] true -> 1:(step'=horizon_step);
endmodule

label "horizon" = step=horizon;

label "accepting" = l=1;
