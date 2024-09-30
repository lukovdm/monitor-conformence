pomdp

observable "i" = s=0;
observable "a" = s=1|s=2;
observable "b" = s=3|s=4;


module main
    s : [0..4] init 0;

    [a] (s=0) -> 0.5:(s'=1) + 0.5:(s'=2);
    [a] (s=1)|(s=2) -> 1:(s'=0);
    [b] (s=1)|(s=2) -> 1:(s'=s+2);
endmodule
