dtmc

module main
    x : [0..2] init 0;

    [] x=0 -> 0.25:(x'=0) + 0.5:(x'=1) + 0.25:(x'=2);
    [] x=1 -> 0.5:(x'=0) + 0.5:(x'=2);
    [] x=2 -> 0.5:(x'=0) + 0.5:(x'=1);
endmodule

label "green" = x>0;
label "red" = x=0;

label "good" = x=0 | x=2;
