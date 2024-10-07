dtmc

const n = 16;

const l1s = 1;
const l1d = 9;
const l2s = 3;
const l2d = 11;
const l3s = 11;
const l3d = 14;

const s1s = 7;
const s1d = 2;
const s2s = 12;
const s2d = 7;
const s3s = 10;
const s3d = 8;

formula ladder = pos=l1s|pos=l2s|pos=l3s;
formula snake = pos=s1s|pos=s2s|pos=s3s;

formula down = pos - ceil(pos/2);
formula up = pos + ceil((n-pos)/2);

module main
    pos : [0..n] init 0;

    [] pos=s1s -> 1:(pos'=s1d);
    [] pos=s2s -> 1:(pos'=s2d);
    [] pos=s3s -> 1:(pos'=s3d);
    [] pos=l1s -> 1:(pos'=l1d);
    [] pos=l2s -> 1:(pos'=l2d);
    [] pos=l3s -> 1:(pos'=l3d);

    [] pos=n -> 1:(pos'=n);
    [] !ladder & !snake & !(pos=n) -> 0.25:(pos'=(pos + 1 > n) ? (n - (n - (pos + 1))*-1) : (pos + 1)) +
                           0.25:(pos'=(pos + 2 > n) ? (n - (n - (pos + 2))*-1) : (pos + 2)) +
                           0.25:(pos'=(pos + 3 > n) ? (n - (n - (pos + 3))*-1) : (pos + 3)) +
                           0.25:(pos'=(pos + 4 > n) ? (n - (n - (pos + 4))*-1) : (pos + 4));
    
endmodule

label "ladder" = ladder;
label "snake" = snake;
label "normal" = !ladder & !snake & pos>0;

label "good" = pos=n;
