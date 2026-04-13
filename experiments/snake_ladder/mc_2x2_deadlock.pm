dtmc

formula ladder = pos=2 & bmec=false;
formula snake = pos=3 & bmec=false;
formula normal = !ladder & !snake & bmec=false;

module main
    pos : [0..4] init 0;

    [step] ladder -> 1:(pos'=pos+2);
    [step] snake -> 1:(pos'=pos-2);
    [step] normal -> 0.5:(pos'=min(4,pos+1)) +
                 0.5:(pos'=min(4,pos+2));
    [step] bmec=true -> 1:(pos'=(pos + 1) % 5);
endmodule

module deadlock
    bmec : bool init false;

    [step] bmec=false -> 0.1:(bmec'=true) + 0.9:(bmec'=false);
    [step] bmec=true -> 1:(bmec'=true);
endmodule

label "b" = bmec=true;
label "ladder" = ladder;
label "snake" = snake;
label "normal" = normal & !(pos=0);

label "good" = pos=4 & bmec=false;
