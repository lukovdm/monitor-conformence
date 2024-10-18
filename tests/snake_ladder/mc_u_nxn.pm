dtmc

const n;

const l1s;
const l1d;
const l2s;
const l2d;
const l3s;
const l3d;
const l4s;
const l4d;
const l5s;
const l5d;
const l6s;
const l6d;
const l7s;
const l7d;
const l8d;
const l8s;
const l9s;
const l9d;
const l10s;
const l10d;

const s1s;
const s1d;
const s2s;
const s2d;
const s3s;
const s3d;
const s4s;
const s4d;
const s5s;
const s5d;
const s6s;
const s6d;
const s7s;
const s7d;
const s8d;
const s8s;
const s9s;
const s9d;
const s10s;
const s10d;

formula ladder = pos=l1s|pos=l2s|pos=l3s|pos=l4s|pos=l5s|pos=l6s|pos=l7s|pos=l8s|pos=l9s|pos=l10s;
formula snake = pos=s1s|pos=s2s|pos=s3s|pos=s4s|pos=s5s|pos=s6s|pos=s7s|pos=s8s|pos=s9s|pos=s10s;

const d = 6;

module main
    pos : [0..n] init 0;

    [] pos=s1s -> 1:(pos'=s1d);
    [] pos=s2s -> 1:(pos'=s2d);
    [] pos=s3s -> 1:(pos'=s3d);
    [] pos=s4s -> 1:(pos'=s4d);
    [] pos=s5s -> 1:(pos'=s5d);
    [] pos=s6s -> 1:(pos'=s6d);
    [] pos=s7s -> 1:(pos'=s7d);
    [] pos=s8s -> 1:(pos'=s8d);
    [] pos=s9s -> 1:(pos'=s9d);
    [] pos=s10s -> 1:(pos'=s10d);

    [] pos=l1s -> 1:(pos'=l1d);
    [] pos=l2s -> 1:(pos'=l2d);
    [] pos=l3s -> 1:(pos'=l3d);
    [] pos=l4s -> 1:(pos'=l4d);
    [] pos=l5s -> 1:(pos'=l5d);
    [] pos=l6s -> 1:(pos'=l6d);
    [] pos=l7s -> 1:(pos'=l7d);
    [] pos=l8s -> 1:(pos'=l8d);
    [] pos=l9s -> 1:(pos'=l9d);
    [] pos=l10s -> 1:(pos'=l10d);

    [] pos=n -> 1:(pos'=n);
    [] !ladder & !snake & !(pos=n) -> 1/d:(pos'=(pos + 1 > n) ? (n - (n - (pos + 1))*-1) : (pos + 1)) +
                                      1/d:(pos'=(pos + 2 > n) ? (n - (n - (pos + 2))*-1) : (pos + 2)) +
                                      1/d:(pos'=(pos + 3 > n) ? (n - (n - (pos + 3))*-1) : (pos + 3)) +
                                      1/d:(pos'=(pos + 4 > n) ? (n - (n - (pos + 4))*-1) : (pos + 4)) +
                                      1/d:(pos'=(pos + 5 > n) ? (n - (n - (pos + 5))*-1) : (pos + 5)) +
                                      1/d:(pos'=(pos + 6 > n) ? (n - (n - (pos + 6))*-1) : (pos + 6));
    
endmodule

label "ladder" = ladder;
label "snake" = snake;
label "normal" = !ladder & !snake & pos>0;

label "good" = pos=n;
