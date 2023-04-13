% setup.m: sets up globals for solution of mixed strategy equilibrium in simultaneous
%          move leapfrogging game
%          Fedor Iskhakov, John Rust, Bertel Schjerning

global bet c0 nstates n cgrid v ir k k1 k2 dtp dt ngp;

dtp=0;      % 1 for deterministic technological progress  (see stp.m)
dt=1;       % interval between moves (usually set this to 1)
c0=5;       % top apex cost state for game
nstates=25; % number of discrete cost states 
bet=.9523;  % discount factor
k=5.2;
k1=5.2;
k2=0;

cgrid=(0:c0/(nstates-1):c0)';
%cgrid=(0:.1/(nstates-1):.1)';

n=nstates*(nstates+1)/2;
v=zeros(n,1);
ir=zeros(n,1);
