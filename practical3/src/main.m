%*****  1D ADVECTION DIFFUSION MODEL OF HEAT TRANSPORT  *******************
%***** 1D ADVECTION DIFFUSION MODEL OF HEAT TRANSPORT  *******************

%***** Initialise Model Setup

% create x-coordinate vectors
xc = dx/2:dx:W-dx/2;    % coordinate vector for cell centre positions [m]
xf = 0:dx:W;            % coordinate vectore for cell face positions [m]

% TASK 4: set time step size [cite: 64]
dt_adv = (dx/2)/u0;                % Courant-Friedrichs-Lewy condition (eq. 7) [cite: 66]
dt_dff = (dx/2)^2/k0;              % Diffusion stability limit (eq. 8) [cite: 69]
dt     = CFL * min(dt_adv,dt_dff); % time step [s] [cite: 72]

% TASK 3: set up ghosted index lists for boundary conditions [cite: 60]
switch BC
    case 'periodic'
        ind3 = [    N,1:N,1  ];        % 3-point stencil
        ind5 = [N-1,N,1:N,1,2];        % 5-point stencil (periodic wrap-around)
    case 'insulating'
        ind3 = [   1,1:N,N   ];        % 3-point stencil
        ind5 = [1,1,1:N,N,N];          % 5-point stencil (mirror boundary)
end

% TASK 2: set initial condition for temperature (eq. 2) [cite: 58, 9]
T   = T0 + dT * exp(-(xc-W/2).^2 / (2*sgm0^2)); 
Tin = T;                                         
Ta  = T;                                         

% Initialise time count variables
t = 0;  
k = 0;  

% initialise output figure
figure(1); clf
makefig(xc,T,Tin,Ta,0);


%***** Solve Model Equations

while t <= tend

    t = t+dt;
    k = k+1;

    % select time integration scheme
    switch TINT
        case 'FE1'  % 1st-order Forward Euler
            dTdt = diffusion(T,k0,dx,ind3) ...
                 + advection(T,u0,dx,ind5,ADVN);

        case 'RK2'  % TASK 6: 2nd-order Runge-Kutta (eq. 6) [cite: 77, 81]
            R1        = diffusion(T,k0,dx,ind3) + advection(T,u0,dx,ind5,ADVN);
            dTdt_half = T + R1*dt/2;
            R2        = diffusion(dTdt_half,k0,dx,ind3) + advection(dTdt_half,u0,dx,ind5,ADVN);
            dTdt      = R2; 
    end

    % update temperature
    T = T + dTdt * dt;

    % TASK 2: get analytical solution at time t (eq. 3 & 4) [cite: 10, 12, 59]
    sgmt = sqrt(sgm0^2 + 2*k0*t); 
    Ta   = T0 + dT * (sgm0/sgmt) * exp(-(xc-W/2-u0*t).^2 / (2*sgmt^2));

    % plot model progress
    if ~mod(k,nop)
        makefig(xc,T,Tin,Ta,t/yr);
        pause(0.1);
    end

end


%***** TASK 8: calculate and display numerical error norm (eq. 5) [cite: 86, 16]
Err = rms(T - Ta, 'all') / rms(Ta, 'all'); 

disp(' ');
disp(['Advection scheme: ',ADVN]);
disp(['Time integration scheme: ',TINT]);
disp(['Numerical error = ',num2str(Err)]);
disp(' ');


%***** Utility Functions  ************************************************

function makefig(x,T,Tin,Ta,t)
    subplot(2,1,1)
    plot(x,Tin,'k:',x,T,'r-',x,Ta,'k--','LineWidth',1.5); axis tight; box on;
    ylabel('T [C]','FontSize',15)
    title(['Temperature at time = ',num2str(t,4),' yr'],'FontSize',18)

    subplot(2,1,2)
    plot(x,(T-Ta)./rms(Ta,'all'),'r-',x,0*Ta,'k-','LineWidth',1.5); axis tight; box on;
    xlabel('x [m]','FontSize',15)
    ylabel('E [1]','FontSize',15)
    title(['Numerical Error at time = ',num2str(t,4),' yr'],'FontSize',18)
    drawnow;
end


%***** TASK 5: Function to calculate diffusion rate [cite: 74]

function dfdt = diffusion(f,k,dx,ind)
    f_padded = f(ind); 
    % calculate diffusive flux q = -k * dT/dx
    q = -k * diff(f_padded) / dx; 
    % calculate diffusion flux balance
    dfdt = -diff(q) / dx; 
end


%*****  Function to calculate advection rate

function dfdt = advection(f,u,dx,ind,ADVN)

% input arguments
% f:    advected scalar field
% u:    advection velocity
% dx:   grid spacing
% ind:  ghosted index list
% ADVN: advection scheme

% output variables
% dfdt: advection rate of scalar field

% split the velocities into positive and negative
u_pos = max(0,u);    % positive velocity (to the right)
u_neg = min(0,u);    % negative velocity (to the left)

% get values on stencil nodes
f_imm  = f(ind(1:end-4));  % i-2
f_im   = f(ind(2:end-3));  % i-1
f_ic   = f(ind(3:end-2));  % i
f_ip   = f(ind(4:end-1));  % i+1
f_ipp  = f(ind(5:end));    % i+2

% get interpolated field values on i+1/2, i-1/2 cell faces
switch ADVN
    case 'UPW1'   % 1st-order upwind scheme
        % positive velocity
        f_ip_pos = f_ic;     % i+1/2
        f_im_pos = f_im;     % i-1/2

        % negative velocity
        f_ip_neg = f_ip;     % i+1/2
        f_im_neg = f_ic;     % i-1/2

    case 'CFD2'  % 2nd-order centred finite-difference scheme
        % positive velocity
        f_ip_pos = (f_ic+f_ip)./2;     % i+1/2
        f_im_pos = (f_ic+f_im)./2;     % i-1/2

        % negative velocity
        f_ip_neg = f_ip_pos;          % i+1/2
        f_im_neg = f_im_pos;          % i-1/2

    case 'UPW3'  % 3rd-order upwind scheme
        % positive velocity
        f_ip_pos = (2*f_ip + 5*f_ic - f_im )./6;     % i+1/2
        f_im_pos = (2*f_ic + 5*f_im - f_imm)./6;     % i-1/2     

        % negative velocity
        f_ip_neg = (2*f_ic + 5*f_ip - f_ipp)./6;     % i+1/2
        f_im_neg = (2*f_im + 5*f_ic - f_ip )./6;     % i-1/2
end

% calculate advection fluxes on i+1/2, i-1/2 cell faces

% positive velocity
q_ip_pos = u_pos.*f_ip_pos;
q_im_pos = u_pos.*f_im_pos;

% negative velocity
q_ip_neg = u_neg.*f_ip_neg;
q_im_neg = u_neg.*f_im_neg;

% advection flux balance for rate of change
div_q_pos = (q_ip_pos - q_im_pos)/dx;  % positive velocity
div_q_neg = (q_ip_neg - q_im_neg)/dx;  % negative velocity

div_q     = div_q_pos + div_q_neg;     % combined
dfdt      = - div_q;                   % advection rate

end
