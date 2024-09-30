dtmc

formula ladder = pos=1|pos=3|pos=6|pos=11;
formula snake = pos=7|pos=12|pos=15;
const n = 16;

formula down = pos - ceil(pos/2);
formula up = pos + ceil((n-pos)/2);

module main
    pos : [0..n] init 0;

    [] ladder -> 1:(pos'=min(n, up));
    [] snake -> 1:(pos'=max(0, down));
    [] !ladder & !snake -> 0.25:(pos'=min(n, pos + 1)) + 
                           0.25:(pos'=min(n, pos + 2)) + 
                           0.25:(pos'=min(n, pos + 3)) + 
                           0.25:(pos'=min(n, pos + 4)); 
    
endmodule

label "ladder" = ladder;
label "snake" = snake;
label "normal" = !ladder & !snake & pos>0;

label "good" = pos=n;
