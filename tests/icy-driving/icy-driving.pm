pomdp

const int IMAX;
formula crash = IMAX + 1;

module car
    pos : [0..IMAX+1] init 0;

    [drive] pos=0 -> 9/10:(pos'=1) + 1/10:(pos'=crash);
    [drive] pos>0 & pos<crash -> 1/4:(pos'=0) + (IMAX-pos+1)/(IMAX*4):(pos'=pos-1) + 
                                (IMAX-pos+1)/(IMAX*4):(pos'=pos) + 
                                (IMAX-pos+1)/(IMAX*4):(pos'=pos+1) + 
                                ((pos-1)*3)/(IMAX*4):(pos'=crash);
    [drive] pos=crash -> 1:(pos'=crash);
endmodule

observable "icy" = pos > 0;

label "crash" = pos=crash;
