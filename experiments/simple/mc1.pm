dtmc

module main
    x : [0..3] init 0;

    [] x=0 -> 0.25:(x'=3) + 0.5:(x'=1) + 0.25:(x'=2);
    [] x=1 -> 0.5:(x'=0) + 0.25:(x'=2) + 0.25:(x'=1);
    [] x=2 -> 0.5:(x'=1) + 0.5:(x'=2);
    [] x=3 -> 0.5:(x'=0) + 0.5:(x'=3);
endmodule

label "green" = x=1 | x=2;
label "red" = x=0 | x=3;

label "good" = x=2;
