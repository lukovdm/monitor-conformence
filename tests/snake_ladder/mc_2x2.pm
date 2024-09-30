dtmc

formula ladder = pos=2;
formula snake = pos=3;
formula normal = !ladder & !snake;

module main
    pos : [0..4] init 0;

    [step] ladder -> 1:(pos'=pos+2);
    [step] snake -> 1:(pos'=pos-2);
    [step] normal -> 0.5:(pos'=min(4,pos+1)) +
                 0.5:(pos'=min(4,pos+2));
endmodule

label "ladder" = ladder;
label "snake" = snake;
label "normal" = normal & !(pos=0);

label "good" = pos=4;
