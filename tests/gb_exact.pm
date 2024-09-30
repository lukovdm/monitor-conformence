mdp

module main
    happy : bool init false;

    [good] happy=false -> (happy'=true);
    [bad] happy=false -> (happy'=false);
    [good] happy=true -> (happy'=true);
    [bad] happy=true -> (happy'=false);
endmodule

label "happy" = happy=true;
