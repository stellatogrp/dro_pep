clear all; clc;

N = 2;
rho = [1.2 1/1.2];
Lf = inf;
mf = 0;
Lg = 1;
mg = .1;
verbose = 1;

% ADMM
%       min_{x,z} f(x) + g(z) st. x+z=c
%
%           x* + z* = c, with 0 \in \partial (f(x*)+g(c-x*))
%
% Saddle point
%       max_{y} min_{x,z} f(x)+g(z) +y^T (x+z-c)
%
% Dual objective
%       min_{y} -f*(-y) - g*(-y) - y^T c
% 
% s_k \in \partial f*(-y_k) <-> -y_k\in\partial f(s_k) <-> f*(-y_k)=-y_k^T s_k - f(s_k) 
%
%
% KKT:    0 == df(x*) + y*    && 0 == dg(z*) + y*
%
%           Lρ(x, z, y) = f(x) + g(z) + y^T (x + z − c) + (ρ/2)‖x + z − c‖2
%
% xk+1 := argminx Lρ(x, zk, yk) // x-minimization
% zk+1 := argminz Lρ(xk+1, z, yk) // z-minimization
% yk+1 := yk + ρ(Axk+1 + Bzk+1 − c) // dual update
%
%
%
% So
%   x_{k+1}: 0\in\partial f(x)+y+rho * (x+z-c)
%       so x_{k+1} = c - z_k - 1/rho *(y_k + \partial f(x_{k+1}))
%   z_{k+1}: 0\in\partial g(z) + y + rho * (x+z-c)
%       so z_{k+1} = c - x_{k+1} - 1/rho * (y_k + \partial g(z_{k+1}))
%   yk+1 = yk + ρ(xk+1 + zk+1 − c)

% PEP formulation?
%
% P = [x0 | z0 | y0 | c | df(x1) ... df(xN) | dg(z1) ... dg(zN) | df(x*) | s_f s_g]
% G = P^T P

dimG = 7 + 2 * N;
dimF = 2 * (N+1);

x0 = zeros(1, dimG); x0(1) = 1; % "x0" = P * x0^T   <<<----> ||x0||^2 = x0*P^T*P*x0^T = tr(G*x0^T x0) 
z0 = zeros(1, dimG); z0(2) = 1;
y0 = zeros(1, dimG); y0(3) = 1;
c = zeros(1, dimG); c(4) = 1;
xs = zeros(1, dimG); zs = c-xs;

df = zeros(N, dimG); df(:,5:4+N) = eye(N);
dg = zeros(N, dimG); dg(:,5+N:4+2*N) = eye(N);
dfs = zeros(1, dimG); dfs(1,5+2*N) = 1;
ys = -dfs;
dgs = - ys;
f = zeros(N, dimF); f(:,1:N) = eye(N); fs = zeros(1,dimF);
g = zeros(N, dimF); g(:,N+1:2*N) = eye(N); gs = zeros(1,dimF);

sf = zeros(1, dimG); sf(1,6+2*N) = 1;
sg = zeros(1, dimG); sg(1,7+2*N) = 1;
fsf = zeros(1, dimF); fsf(1,2*N+1) = 1;
gsg = zeros(1, dimF); gsg(1,2*N+2) = 1;

xk = zeros(N+1,dimG); xk(1,:) = x0;
zk = zeros(N+1,dimG); zk(1,:) = z0;
yk = zeros(N+1,dimG); yk(1,:) = y0;
for i = 1:N
    xk(i+1,:) = c - zk(i,:) - 1/rho(i) * ( yk(i,:) + df(i,:) );
    zk(i+1,:) = c - xk(i+1,:) - 1/rho(i) * ( yk(i,:) + dg(i,:) );
    yk(i+1,:) = yk(i,:) + rho(i) * (xk(i+1,:)+zk(i+1,:)-c);
end

XF = [xs; xk(2:end,:);sf];
GF = [dfs; df;-yk(end,:)];
FF = [fs; f; fsf];
XG = [zs; zk(2:end,:);sg];
GG = [dgs; dg; -yk(end,:)];
FG = [gs; g; gsg];

nbPtsf = 2 + N;
nbPtsg = 2 + N;

%%

lambda_g = sdpvar(nbPtsg,nbPtsg,'full'); cons = (lambda_g >= 0);
lambda_f = sdpvar(nbPtsf,nbPtsf,'full'); cons = cons + (lambda_f >= 0);
tau = sdpvar(1); cons = cons + (tau>=0);

% G = sdpvar(dimG); cons = (G>=0);
% F = sdpvar(dimF,1);

% cons = cons + ( (x0-xs)*G*(x0-xs)'  <= 1 );
% cons = cons + ( (y0-ys)*G*(y0-ys)'  <= 1 );
% cons = cons + ( (z0-zs)*G*(z0-zs)'  <= 1 );
% cons = cons + ( (x0+z0-c)*G*(x0+z0-c)'  <= 1 );
% cons = cons + ( (z0-zs)*G*(z0-zs)'  + (y0-ys)*G*(y0-ys)' <= 1 ); 

PSD_dual = tau * ( (z0-zs)'*(z0-zs)  + (y0-ys)'*(y0-ys));
LIN_dual = 0 * fs;

m = mg; L = Lg;
for i = 1:nbPtsg
    for j = 1:nbPtsg
        if i ~= j
            xi = XG(i,:); xj = XG(j,:);
            gi = GG(i,:); gj = GG(j,:);
            fi = FG(i,:); fj = FG(j,:);
            if L ~= inf
%                 cons = cons + (0 >= (fj-fi)*F + gj*G*(xi-xj)' + 1/2/L * (gi-gj)*G*(gi-gj)' + m/2/(1-m/L) * (xi-xj-1/L*(gi-gj))*G*(xi-xj-1/L*(gi-gj))' );
                bij = (fj-fi);
                Aij = gj'*(xi-xj) + 1/2/L * (gi-gj)'*(gi-gj) + m/2/(1-m/L) * (xi-xj-1/L*(gi-gj))'*(xi-xj-1/L*(gi-gj));
            else
%                 cons = cons + (0 >= (fj-fi)*F + gj*G*(xi-xj)' + m/2 * (xi-xj)*G*(xi-xj)' );
                bij = (fj-fi);
                Aij = gj'*(xi-xj) + m/2 * (xi-xj)'*(xi-xj);
            end
            Aij = 1/2 * (Aij+Aij');
            PSD_dual = PSD_dual + lambda_g(i,j) * Aij;
            LIN_dual = LIN_dual + lambda_g(i,j) * bij;
        end
    end
end

m = mf; L = Lf;
for i = 1:nbPtsf
    for j = 1:nbPtsf
        if i ~= j
            xi = XF(i,:); xj = XF(j,:);
            gi = GF(i,:); gj = GF(j,:);
            fi = FF(i,:); fj = FF(j,:);
            if L ~= inf
%                 cons = cons + (0 >= (fj-fi)*F + gj*G*(xi-xj)' + 1/2/L * (gi-gj)*G*(gi-gj)' + m/2/(1-m/L) * (xi-xj-1/L*(gi-gj))*G*(xi-xj-1/L*(gi-gj))' );
                bij = (fj-fi);
                Aij = gj'*(xi-xj) + 1/2/L * (gi-gj)'*(gi-gj) + m/2/(1-m/L) * (xi-xj-1/L*(gi-gj))'*(xi-xj-1/L*(gi-gj));
            else
%                 cons = cons + (0 >= (fj-fi)*F + gj*G*(xi-xj)' + m/2 * (xi-xj)*G*(xi-xj)' );
                bij = (fj-fi);
                Aij = gj'*(xi-xj) + m/2 * (xi-xj)'*(xi-xj);
            end
            Aij = 1/2 * (Aij+Aij');
            PSD_dual = PSD_dual + lambda_f(i,j) * Aij;
            LIN_dual = LIN_dual + lambda_f(i,j) * bij;
        end
    end
end
% Residual
% obj = (xk(end,:)+zk(end,:)-c)*G*(xk(end,:)+zk(end,:)-c)';
% Dual objective
% obj = (fs+gs)*F + (-yk(end,:)*G*sf' - fsf*F) + (-yk(end,:)*G*sg'- gsg*F) + yk(end,:)*G*c';
% Distance

Aij_obj = (zk(end,:)-zs)'*(zk(end,:)-zs)  + (yk(end,:)-ys)'*(yk(end,:)-ys);
Aij_obj = 1/2 * (Aij_obj + Aij_obj');
PSD_dual = PSD_dual - Aij_obj ;
% obj = (zk(end,:)-zs)*G*(zk(end,:)-zs)'  + (yk(end,:)-ys)*G*(yk(end,:)-ys)' 

cons = cons + (PSD_dual >= 0);
cons = cons + (LIN_dual == 0);
options = sdpsettings('verbose',verbose,'solver','mosek');

status = optimize(cons,tau,options);


 [double(tau)]
















