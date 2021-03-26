%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------- Predictive global neuronal workspace simulation -------%
%------------------------ Taxonomy & IB ------------------------%
%------------- Christopher Whyte & Ryan Smith 2020 -------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
rng('shuffle')

%--------------------% Simulation settings ---------------------%
% To simulate sustained inattentional blindness and the report b-
% ehvaiour described in the taxonomy set TAX = 1; to simulate the 
% 2AF behaviour described in the taxonomy set TAX = 2.

% Allow some minutes for the simulations to run

TAX = 1;   
    
for z = 1:5
%% task settings

att_off = 0.05; %attention off
att_on = 1; %attention on
ign = .6; %2nd level A matrix precision

    if z == 1 %subliminal (unattended)
        evi = 0.01;
    elseif z == 2 %subliminal (attended)
        evi = 0.01;
    elseif z == 3 %supraliminal (unattended)
        evi = .7;
    elseif z == 4 %supraliminal attended
        evi = .7;
    elseif z == 5 %Phase 1: sustained inattentional blindness
        evi = .75;
    end 

%% Level 1
%==========================================================================

% prior beliefs about initial states
%--------------------------------------------------------------------------

D{1} = [1 1]';% inside stim shape {Square, Random}
D{2} = [1 1]';% colour of outside stim {Black, Red}
D{3} = [0 0]';% attention {square, only color}
D{4} = [1 0 0 0 0 0 0]';% language processing {silent, "I", "see" "a", "square", "didn't", "anything"}


% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------

% Outcome modality 1:shape
for i = 1:2
    for j = 1:2
        for k = 1:7
        A{1}(:,:,i,j,k) = [1 0;%square
                           0 1];%random
        end
    end
end

                  
% Outcome modality 2:colour

for k = 1:7
    A{2}(:,:,1,1,k) = [1 1;% black
                       0 0];% red
    A{2}(:,:,1,2,k) = [1 1;% black
                       0 0];% red
    A{2}(:,:,2,1,k) = [0 0;
                       1 1]; 
    A{2}(:,:,2,2,k) = [0 0;
                       1 1];
end

for i = 1:2
    for j = 1:2
        for k = 1:7
        A{3}(k,:,i,j,k) = [1 1];
        end
    end
end
             
a = A;


%evidence accumulation
a{1}(:,:,:,:,:) = spm_softmax(evi*log(a{1}(:,:,:,:,:)+exp(-4))); % attenuate precision of mapping to shape

%attention
if z == 1 
    for i = 1:2
        a{1}(:,:,i,2,:) = spm_softmax(att_off*log(a{1}(:,:,i,2,:)+exp(-4))); % attenuate precision of mapping to shape
    end 
elseif z == 2
    for i = 1:2
        a{1}(:,:,i,1,:) = spm_softmax(att_on*log(a{1}(:,:,i,1,:)+exp(-4))); % boost precision of mapping to shape
    end 
elseif z == 3 
    for i = 1:2
        a{1}(:,:,i,2,:) = spm_softmax(att_off*log(a{1}(:,:,i,2,:)+exp(-4))); % attenuate precision of mapping to shape
    end 
elseif z == 4
    for i = 1:2
        a{1}(:,:,i,1,:) = spm_softmax(att_on*log(a{1}(:,:,i,1,:)+exp(-4))); % boost precision of mapping to shape
    end 
end 

%seperate generative process from generative model (multiplying by 64
%prevents learning by making concentration parameteres very high).

a{1}= a{1}*64;
a{2} = a{2}*64;
a{3} = a{3}*64;

% Transitions between states: B
%--------------------------------------------------------------------------

B{1}= eye(2,2);

B{2}= eye(2,2);

B{3}= eye(2,2);

B{4}= eye(7,7);
 
% MDP Structure
%--------------------------------------------------------------------------
mdp.T = 1;                      % number of updates
mdp.A = A;                      % observation model
mdp.B = B;                      % transition probabilities
mdp.D = D;                      % prior over initial states
mdp.a = a;
mdp.erp = 1; 

mdp.Aname = {'Shape', 'Color','word'};
mdp.Bname = {'Shape', 'Color', 'attention','word'};
 
clear A B D
 
MDP = spm_MDP_check(mdp);

clear mdp

%% Level 2 (slower semantic timescale)
%==========================================================================
 
% prior beliefs about initial states (in terms of counts_: D and d
%--------------------------------------------------------------------------
D{1} = [1 1 1 1]';% Sequence type: {black+square, black+random, red+square, red+random}
D{2} = [1 0 0 0 0 0 0 0]'; % time in trial: {Start, change1,2, return, remember1,2,3,4}
D{3} = [1 0 0]'; % Report: {Null, unseen, seen}
d = D;

% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------

for i = 1:8
    for j = 1:3
    A{1}(:,:,i,j) = [0 0 0 0;%square
                    1 1 1 1];%random
    end
end

for i = 2:3
    for j = 1:3
    A{1}(:,:,i,j) = [1 0 1 0;
                     0 1 0 1];
    end
end


for i = 1:8
    for j = 1:3
    A{2}(:,:,i,j) = [1 1 0 0;%black
                     0 0 1 1];%red
    end
end


              
for i = 1:8
    for j = 1:3
A{3}(:,:,i,j) = [1 1 1 1;%silent
                 0 0 0 0;%"I"
                 0 0 0 0;%"see"
                 0 0 0 0;% "a"
                 0 0 0 0;%"square"
                 0 0 0 0;%"didn't"
                 0 0 0 0];%"anything"
    end
end

%unseen

A{3}(:,:,5,2) = [0 0 0 0;%silent
                 1 1 1 1;%"I"
                 0 0 0 0;%"see"
                 0 0 0 0;% "a"
                 0 0 0 0;%"square"
                 0 0 0 0;%"didn't"
                 0 0 0 0];%"anything"
             
A{3}(:,:,6,2) = [0 0 0 0;%silent
                 0 0 0 0;%"I"
                 0 0 0 0;%"see"
                 0 0 0 0;% "a"
                 0 0 0 0;%"square"
                 1 1 1 1;%"didn't"
                 0 0 0 0];%"anything"
A{3}(:,:,7,2) = [0 0 0 0;%silent
                 0 0 0 0;%"I"
                 1 1 1 1;%"see"
                 0 0 0 0;% "a"
                 0 0 0 0;%"square"
                 0 0 0 0;%"didn't"
                 0 0 0 0];%"anything"
A{3}(:,:,8,2) = [0 0 0 0;%silent
                 0 0 0 0;%"I"
                 0 0 0 0;%"see"
                 0 0 0 0;% "a"
                 0 0 0 0;%"square"
                 0 0 0 0;%"didn't"
                 1 1 1 1];%"anything"
             
%seen
             
A{3}(:,:,5,3) = [0 0 0 0;%silent
                 1 1 1 1;%"I"
                 0 0 0 0;%"see"
                 0 0 0 0;% "a"
                 0 0 0 0;%"square"
                 0 0 0 0;%"didn't"
                 0 0 0 0];%"anything"
             
 A{3}(:,:,6,3) = [0 0 0 0;%silent
                 0 0 0 0;%"I"
                 1 1 1 1;%"see"
                 0 0 0 0;% "a"
                 0 0 0 0;%"square"
                 0 0 0 0;%"didn't"
                 0 0 0 0];%"anything"
 A{3}(:,:,7,3) = [0 0 0 0;%silent
                 0 0 0 0;%"I"
                 0 0 0 0;%"see"
                 1 1 1 1;% "a"
                 0 0 0 0;%"square"
                 0 0 0 0;%"didn't"
                 0 0 0 0];%"anything"
 A{3}(:,:,8,3) = [0 0 0 0;%silent
                 0 0 0 0;%"I"
                 0 0 0 0;%"see"
                 0 0 0 0;% "a"
                 1 1 1 1;%"square"
                 0 0 0 0;%"didn't"
                 0 0 0 0];%"anything"


for i = 1:8
    for j = 1:3
    A{4}(:,:,i,j) = [1 1 1 1;%null
                     0 0 0 0;%incorrect
                     0 0 0 0];%correct
    end
end

%unseen
if TAX == 2
    for i = 5:8
        A{4}(:,:,i,2) = [0 0 0 0;%null
                         1 0 1 0;%incorrect
                         0 1 0 1];%correct
    end
end

%seen
for i = 5:8
    A{4}(:,:,i,3) = [0 0 0 0;%null
                     0 1 0 1;%incorrect
                     1 0 1 0];%correct
end
             
a2 = A;

for i = 2    
    for j = 1:3
        a2{1}(:,:,i,j) = spm_softmax(ign*log(a2{1}(:,:,i,j)+exp(-4))); %decrease precision of mapping to shape
    end
end

%seperate generative process from generative model (multiplying by 64
%prevents learning by making concentration parameteres very high).
a2{1}= a2{1}*64;
a2{2} = a2{2}*64;
a2{3} = a2{3}*64;
a2{4} = a2{4}*64;

%transitions: B
%--------------------------------------------------------------------------
B{1} = eye(4,4);

B{2} = [0 0 0 0 0 0 0 0;
        1 0 0 0 0 0 0 0;
        0 1 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0;
        0 0 0 1 0 0 0 0;
        0 0 0 0 1 0 0 0;
        0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 1 1];

    
B{3}(:,:,1) = [1 1 1;
               0 0 0;
               0 0 0];          
B{3}(:,:,2) = [0 0 0;
               1 1 1;
               0 0 0];
B{3}(:,:,3) = [0 0 0;
               0 0 0;
               1 1 1];

% Policies
%--------------------------------------------------------------------------

T = 8; %number of timesteps
Nf = 3; %number of factors
Pi = 2; %number of policies
V = ones(T-1,Pi,Nf);

V(:,:,3) = [1 1;
            1 1;
            1 1;
            2 3;
            2 3;
            2 3;
            2 3]; 
 
% C matrices (outcome modality by timesteps)
%--------------------------------------------------------------------------

C{1} = zeros(2,8);
    
C{2} = zeros(2,8);
    
C{3} = zeros(7,8);
    
%confidence/report
if TAX == 1
    i = -12;
    c = 3;
elseif TAX == 2
    i = -1;
    c = 1;
end 

C{4} = [0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 i;
        0 0 0 0 0 0 0 c];
             
% MDP Structure
%--------------------------------------------------------------------------
mdp.MDP  = MDP;
mdp.link = [1 0 0 0;
            0 1 0 0;
            0 0 0 0;
            0 0 1 0];

mdp.T = T;                      % number of updates
mdp.A = A;                      % observation model
mdp.B = B;                      % transition probabilities
mdp.C = C;                      % preferred outcomes
mdp.D = D;                      % prior over initial states
mdp.V = V;                      % policies
mdp.s = 1;
mdp.a = a2;
mdp.d = d;
mdp.beta = 4;                   %hyperprior on policy
mdp.alpha = 6;                  %motor stochasity 
mdp.erp = 1;


mdp.Aname = {'Shape', 'Colour', 'word','Report feedback'};
mdp.Bname = {'Sequence type', 'Time in trial', 'Report'};


label.modality{3} = 'self-report';    label.outcome{3} = {'silent', 'I', 'see' "a", 'square', 'didn''t', 'anything'};

mdp.label = label;

if z == 1 
    mdp_1 = spm_MDP_check(mdp);
elseif z ==2 
    mdp_2 = spm_MDP_check(mdp);
elseif z ==3 
    mdp_3 = spm_MDP_check(mdp);
elseif z ==4
    mdp_4 = spm_MDP_check(mdp);
elseif z ==5 && TAX ==1
    mdp_SIB = spm_MDP_check(mdp);
end 

clear A B C D V
clear mdp
clear MDP

end 

%% Construct MDP Structures
%==========================================================================

if TAX == 1
    
    % Taxonomy: report
    %======================================================================
    
    % Subliminal no attention
    %----------------------------------------------------------------------
    
    %unexpected
    rng('shuffle')
    for i = 1:100
        MDP_1 = mdp_1;
        MDP_1.d{1} = [1 1.5 1 1.5]'*64;
        MDP_1.MDP.D{3} = [0 1]';
        MDP_1l{i} = spm_MDP_VB_X_PGNW(MDP_1);
    end 
    
    for i = 1:100
        state(i) = MDP_1l{i}.s(3,8);
        acc(i) = MDP_1l{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            count1l(i) = 1;
        else 
            count1l(i) = 0;
        end 
    end 

    count1l = sum(count1l)
    
    
    %neutral
    rng('shuffle')

    for i = 1:100
        MDP_1 = mdp_1;
        MDP_1.d{1} = [1 1 1 1]'*64;
        MDP_1.MDP.D{3} = [0 1]';
        MDP_1n{i} = spm_MDP_VB_X_PGNW(MDP_1);
    end 
    
    for i = 1:100
        state(i) = MDP_1n{i}.s(3,8);
        acc(i) = MDP_1n{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            count1n(i) = 1;
        else 
            count1n(i) = 0;
        end 
    end 

    count1n = sum(count1n)
    
    %expected
    rng('shuffle')

    for i = 1:100
        MDP_1 = mdp_1;
        MDP_1.d{1} = [1.5 1 1.5 1]'*64;
        MDP_1.MDP.D{3} = [0 1]';
        MDP_1h{i} = spm_MDP_VB_X_PGNW(MDP_1);
    end 
    
    for i = 1:100
        state(i) = MDP_1h{i}.s(3,8);
        acc(i) = MDP_1h{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            count1h(i) = 1;
        else 
            count1h(i) = 0;
        end 
    end 

    count1h = sum(count1h)
    
    
    % Subliminal attended
    %----------------------------------------------------------------------
    
    %unexpected
    rng('shuffle')    
    for i = 1:100
        MDP_2 = mdp_2;
        MDP_2.d{1} = [1 1.5 1 1.5]'*64;
        MDP_2.MDP.D{3} = [1 0]'; 
        MDP_2l{i} = spm_MDP_VB_X_PGNW(MDP_2);
    end 
    
    for i = 1:100
        state(i) = MDP_2l{i}.s(3,8);
        acc(i) = MDP_2l{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            count2l(i) = 1;
        else 
            count2l(i) = 0;
        end 
    end 

    count2l = sum(count2l)
    
    %neutral
    rng('shuffle')    
    for i = 1:100
        MDP_2 = mdp_2;
        MDP_2.d{1} = [1 1 1 1]'*64;
        MDP_2.MDP.D{3} = [1 0]'; 
        MDP_2n{i} = spm_MDP_VB_X_PGNW(MDP_2);
    end 
    
    for i = 1:100
        state(i) = MDP_2n{i}.s(3,8);
        acc(i) = MDP_2n{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            count2n(i) = 1;
        else 
            count2n(i) = 0;
        end 
    end 

    count2n = sum(count2n)
    
    %expected
    rng('shuffle')    
    for i = 1:100
        MDP_2 = mdp_2;
        MDP_2.d{1} = [1.5 1 1.5 1]'*64;
        MDP_2.MDP.D{3} = [1 0]'; 
        MDP_2h{i} = spm_MDP_VB_X_PGNW(MDP_2);
    end 
    
    for i = 1:100
        state(i) = MDP_2h{i}.s(3,8);
        acc(i) = MDP_2h{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            count2h(i) = 1;
        else 
            count2h(i) = 0;
        end 
    end 

    count2h = sum(count2h)
    
    % Preconscious
    %----------------------------------------------------------------------
    
    %unexpected
    rng('shuffle')
    for i = 1:100
        MDP_3 = mdp_3;
        MDP_3.d{1} = [1 1.5 1 1.5]'*64;
        MDP_3.MDP.D{3} = [0 1]';
        MDP_3l{i} = spm_MDP_VB_X_PGNW(MDP_3);
    end 
    
    for i = 1:100
        state(i) = MDP_3l{i}.s(3,8);
        acc(i) = MDP_3l{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            count3l(i) = 1;
        else 
            count3l(i) = 0;
        end 
    end 

    count3l = sum(count3l)
    
    %neutral
    rng('shuffle')
    for i = 1:100
        MDP_3 = mdp_3;
        MDP_3.d{1} = [1 1 1 1]'*64;
        MDP_3.MDP.D{3} = [0 1]';
        MDP_3n{i} = spm_MDP_VB_X_PGNW(MDP_3);
    end 
    
    for i = 1:100
        state(i) = MDP_3n{i}.s(3,8);
        acc(i) = MDP_3n{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            count3n(i) = 1;
        else 
            count3n(i) = 0;
        end 
    end 

    count3n = sum(count3n)
    
    %expected
    rng('shuffle')
    for i = 1:100
        MDP_3 = mdp_3;
        MDP_3.d{1} = [1.5 1 1.5 1]'*64;
        MDP_3.MDP.D{3} = [0 1]';
        MDP_3h{i} = spm_MDP_VB_X_PGNW(MDP_3);
    end 
    
    for i = 1:100
        state(i) = MDP_3h{i}.s(3,8);
        acc(i) = MDP_3h{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            count3h(i) = 1;
        else
            count3h(i) = 0;
        end 
    end 

    count3h = sum(count3h)

    
    % Conscious
    %----------------------------------------------------------------------
    
    %unexpected
    rng('shuffle')
    for i = 1:100
        MDP_4 = mdp_4;
        MDP_4.d{1} = [1 1.5 1 1.5]'*64;
        MDP_4.MDP.D{3} = [1 0]';
        MDP_4l{i} = spm_MDP_VB_X_PGNW(MDP_4);
    end 
    
    for i = 1:100
        state(i) = MDP_4l{i}.s(3,8);
        acc(i) = MDP_4l{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            count4l(i) = 1;
        else 
            count4l(i) = 0;
        end 
    end 

    count4l = sum(count4l)
    
    %neutral
    rng('shuffle')
    for i = 1:100
        MDP_4 = mdp_4;
        MDP_4.d{1} = [1 1 1 1]'*64;
        MDP_4.MDP.D{3} = [1 0]';
        MDP_4n{i} = spm_MDP_VB_X_PGNW(MDP_4);
    end 
    
    for i = 1:100
        state(i) = MDP_4n{i}.s(3,8);
        acc(i) = MDP_4n{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            count4n(i) = 1;
        else 
            count4n(i) = 0;
        end 
    end 

    count4n = sum(count4n)
    
    %expected
    rng('shuffle')
    for i = 1:100
        MDP_4 = mdp_4;
        MDP_4.d{1} = [1.5 1 1.5 1]'*64;
        MDP_4.MDP.D{3} = [1 0]';
        MDP_4h{i} = spm_MDP_VB_X_PGNW(MDP_4);
    end 
    
    for i = 1:100
        state(i) = MDP_4h{i}.s(3,8);
        acc(i) = MDP_4h{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            count4h(i) = 1;
        else 
            count4h(i) = 0;
        end 
    end 

    count4h = sum(count4h)
    
    %Sustained inattentional blindness
    %======================================================================
    %Phase 1
    %----------------------------------------------------------------------
    rng('shuffle')
    for i = 1:100
        MDP_P11 = mdp_SIB;
        MDP_P11.d{1} = [1 1 1 1]'*64;
        MDP_P11.MDP.D{3} = [0 1]';
        %ATTENTION: decrease precision of shape outcome modality
        MDP_P11.MDP.a{1}(:,:,:,2) = spm_softmax(att_off*log(a{1}(:,:,:,2)+exp(-4))); %decrease precision of mapping to shape 
        MDP_P11 = spm_MDP_check(MDP_P11);
        MDP_P1{i} = spm_MDP_VB_X_PGNW(MDP_P11);
    end 

    for i = 1:100
        state(i) = MDP_P1{i}.s(3,8);
        acc(i) = MDP_P1{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            countP1(i) = 1;
        else 
            countP1(i) = 0;
        end 
    end 

    countP1 = sum(countP1)
    
    % Phase 2
    %----------------------------------------------------------------------
    rng('shuffle')
    for i = 1:100
        MDP_P22 = mdp_SIB;
        MDP_P22.d{1} = [1 1 1 1]'*64;
        MDP_P22.MDP.D{3} = [0 1]';
        %ATTENTION: decrease precision of shape outcome modality
        MDP_P22.MDP.a{1}(:,:,:,2) = spm_softmax(.2*log(a{1}(:,:,:,2)+exp(-4))); %decrease precision of mapping to shape 
        MDP_P22 = spm_MDP_check(MDP_P22);
        MDP_P2{i} = spm_MDP_VB_X_PGNW(MDP_P22);
    end 
        
    for i = 1:100
        state(i) = MDP_P2{i}.s(3,8);
        acc(i) = MDP_P2{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            countP2(i) = 1;
        else 
            countP2(i) = 0;
        end 
    end  

    countP2 = sum(countP2)
    
    % Phase 3
    %----------------------------------------------------------------------
    rng('shuffle')
    for i = 1:100
        MDP_P33 = mdp_SIB;
        MDP_P33.d{1} = [1 1 1 1]'*64;
        MDP_P33.MDP.D{3} = [1 0]';
        %ATTENTION: boost precision of shape outcome modality
        MDP_P33.MDP.a{1}(:,:,:,2) = spm_softmax(att_on*log(a{1}(:,:,:,2)+exp(-4))); % attenuate precision of mapping to shape
        MDP_P33 = spm_MDP_check(MDP_P33);
        MDP_P3{i} = spm_MDP_VB_X_PGNW(MDP_P33);
    end 
    
    for i = 1:100
        state(i) = MDP_P3{i}.s(3,8);
        acc(i) = MDP_P3{i}.o(4,5);
    end 

    for i = 1:100
        if acc(i) == 3 && state(i) == 3
            countP3(i) = 1;
        else 
            countP3(i) = 0;
        end 
    end 
    
    countP3 = sum(countP3)
    
    
elseif TAX ==2
    
    % Taxonomy: 2AF
    %======================================================================
    
    
    % Taxonomy: report
    %======================================================================
    
    % Subliminal no attention
    %----------------------------------------------------------------------
    
    %unexpected
    rng('shuffle')
    for i = 1:100
        MDP_1 = mdp_1;
        MDP_1.d{1} = [1 1.5 1 1.5]'*64;
        MDP_1.MDP.D{3} = [0 1]';
        MDP_1l{i} = spm_MDP_VB_X_PGNW(MDP_1);
    end 
    
    for i = 1:100
        acc(i) = MDP_1l{i}.o(4,5);
    end 

    for i = 1:100
            count1l(i) = acc(i) == 3;
    end 

    count1l = sum(count1l)
       
    %neutral
    rng('shuffle')

    for i = 1:100
        MDP_1 = mdp_1;
        MDP_1.d{1} = [1 1 1 1]'*64;
        MDP_1.MDP.D{3} = [0 1]';
        MDP_1n{i} = spm_MDP_VB_X_PGNW(MDP_1);
    end 
    
    for i = 1:100
        acc(i) = MDP_1n{i}.o(4,5);
    end 

    for i = 1:100
            count1n(i) = acc(i) == 3;
    end 

    count1n = sum(count1n)
    
    %expected
    rng('shuffle')

    for i = 1:100
        MDP_1 = mdp_1;
        MDP_1.d{1} = [1.5 1 1.5 1]'*64;
        MDP_1.MDP.D{3} = [0 1]';
        MDP_1h{i} = spm_MDP_VB_X_PGNW(MDP_1);
    end 
    
    for i = 1:100
        acc(i) = MDP_1h{i}.o(4,5);
    end 

    for i = 1:100
            count1h(i) = acc(i) == 3;
    end 

    count1h = sum(count1h)
    
    
    % Subliminal attended
    %----------------------------------------------------------------------
    
    %unexpected
    rng('shuffle')    
    for i = 1:100
        MDP_2 = mdp_2;
        MDP_2.d{1} = [1 1.5 1 1.5]'*64;
        MDP_2.MDP.D{3} = [1 0]'; 
        MDP_2l{i} = spm_MDP_VB_X_PGNW(MDP_2);
    end 
    
    for i = 1:100
        acc(i) = MDP_2l{i}.o(4,5);
    end 

    for i = 1:100
            count2l(i) = acc(i) == 3;
    end 

    count2l = sum(count2l)
 
    
    %neutral
    rng('shuffle')    
    for i = 1:100
        MDP_2 = mdp_2;
        MDP_2.d{1} = [1 1 1 1]'*64;
        MDP_2.MDP.D{3} = [1 0]'; 
        MDP_2n{i} = spm_MDP_VB_X_PGNW(MDP_2);
    end 
    
    for i = 1:100
        acc(i) = MDP_2n{i}.o(4,5);
    end 

    for i = 1:100
            count2n(i) = acc(i) == 3;
    end 

    count2n = sum(count2n)
    
    %expected
    rng('shuffle')    
    for i = 1:100
        MDP_2 = mdp_2;
        MDP_2.d{1} = [1.5 1 1.5 1]'*64;
        MDP_2.MDP.D{3} = [1 0]'; 
        MDP_2h{i} = spm_MDP_VB_X_PGNW(MDP_2);
    end 
    
    for i = 1:100
        acc(i) = MDP_2h{i}.o(4,5);
    end 

    for i = 1:100
            count2h(i) = acc(i) == 3;
    end 

    count2h = sum(count2h)
       
    
    % Preconscious
    %----------------------------------------------------------------------
    
    %unexpected
    rng('shuffle')
    for i = 1:100
        MDP_3 = mdp_3;
        MDP_3.d{1} = [1 1.5 1 1.5]'*64;
        MDP_3.MDP.D{3} = [0 1]';
        MDP_3l{i} = spm_MDP_VB_X_PGNW(MDP_3);
    end 
    
    for i = 1:100
        acc(i) = MDP_3l{i}.o(4,5);
    end 

    for i = 1:100
            count3l(i) = acc(i) == 3;
    end 

    count3l = sum(count3l)
       
    
    %neutral
    rng('shuffle')
    for i = 1:100
        MDP_3 = mdp_3;
        MDP_3.d{1} = [1 1 1 1]'*64;
        MDP_3.MDP.D{3} = [0 1]';
        MDP_3n{i} = spm_MDP_VB_X_PGNW(MDP_3);
    end 
    
    for i = 1:100
        acc(i) = MDP_3n{i}.o(4,5);
    end 

    for i = 1:100
            count3n(i) = acc(i) == 3;
    end 

    count3n = sum(count3n)
       
    
    %expected
    rng('shuffle')
    for i = 1:100
        MDP_3 = mdp_3;
        MDP_3.d{1} = [1.5 1 1.5 1]'*64;
        MDP_3.MDP.D{3} = [0 1]';
        MDP_3h{i} = spm_MDP_VB_X_PGNW(MDP_3);
    end 
    
    for i = 1:100
        acc(i) = MDP_3h{i}.o(4,5);
    end 

    for i = 1:100
            count3h(i) = acc(i) == 3;
    end 

    count3h = sum(count3h)
       

    
    % Conscious
    %----------------------------------------------------------------------
    
    %unexpected
    rng('shuffle')
    for i = 1:100
        MDP_4 = mdp_4;
        MDP_4.d{1} = [1 1.5 1 1.5]'*64;
        MDP_4.MDP.D{3} = [1 0]';
        MDP_4l{i} = spm_MDP_VB_X_PGNW(MDP_4);
    end 
    
    for i = 1:100
        acc(i) = MDP_4l{i}.o(4,5);
    end 

    for i = 1:100
            count4l(i) = acc(i) == 3;
    end 

    count4l = sum(count4l)
       
    
    %neutral
    rng('shuffle')
    for i = 1:100
        MDP_4 = mdp_4;
        MDP_4.d{1} = [1 1 1 1]'*64;
        MDP_4.MDP.D{3} = [1 0]';
        MDP_4n{i} = spm_MDP_VB_X_PGNW(MDP_4);
    end 
    
    for i = 1:100
        acc(i) = MDP_4n{i}.o(4,5);
    end 

    for i = 1:100
            count4n(i) = acc(i) == 3;
    end 

    count4n = sum(count4n)
       
    
    %expected
    rng('shuffle')
    for i = 1:100
        MDP_4 = mdp_4;
        MDP_4.d{1} = [1.5 1 1.5 1]'*64;
        MDP_4.MDP.D{3} = [1 0]';
        MDP_4h{i} = spm_MDP_VB_X_PGNW(MDP_4);
    end 
    
    for i = 1:100
        acc(i) = MDP_4h{i}.o(4,5);
    end 

    for i = 1:100
            count4h(i) = acc(i) == 3;
    end 

    count4h = sum(count4h)
    
%     avg = [count1l count1n count1h;
%            count2l count2n count2h;
%            count3l count3n count3h;
%            count4l count4n count4h;
%            countP1 countP2 countP3]
%        
%     save('avg1', 'avg')
%           
end 


%% Plot
%==========================================================================

if TAX == 1
    
    % Taxonomy: report
    %======================================================================
    
    spm_figure('GetWin','Subliminal, unattended, low prior: trial'); clf
    spm_MDP_VB_trial(MDP_1l{1},1:3,[3]);
    spm_figure('GetWin','Subliminal, unattended, low prior: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_1l{1});
    spm_figure('GetWin','Subliminal, unattended, low prior: LFP'); clf
    spm_MDP_VB_LFP(MDP_1l{1},[],1);
    spm_figure('GetWin','Subliminal, unattended, neutral: trial'); clf
    spm_MDP_VB_trial(MDP_1n{1},1:3,[3]);
    spm_figure('GetWin','Subliminal, unattended, neutral: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_1n{1});
    spm_figure('GetWin','Subliminal, unattended, neutral: LFP'); clf
    spm_MDP_VB_LFP(MDP_1n{1},[],1);
    spm_figure('GetWin','Subliminal, unattended, high prior: trial'); clf
    spm_MDP_VB_trial(MDP_1h{1},1:3,[3]);
    spm_figure('GetWin','Subliminal, unattended, high prior: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_1h{1});
    spm_figure('GetWin','Subliminal, unattended, high prior: LFP'); clf
    spm_MDP_VB_LFP(MDP_1h{1},[],1);
    spm_figure('GetWin','Subliminal, attended, low prior: trial'); clf
    spm_MDP_VB_trial(MDP_2l{1},1:3,[3]);
    spm_figure('GetWin','Subliminal, attended, low prior: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_2l{1});
    spm_figure('GetWin','Subliminal, attended, low prior: LFP'); clf
    spm_MDP_VB_LFP(MDP_2l{1},[],1);
    spm_figure('GetWin','Subliminal, attended, neutral: trial'); clf
    spm_MDP_VB_trial(MDP_2n{1},1:3,[3]);
    spm_figure('GetWin','Subliminal, attended, neutral: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_2n{1});
    spm_figure('GetWin','Subliminal, attended, neutral: LFP'); clf
    spm_MDP_VB_LFP(MDP_2n{1},[],1);
    spm_figure('GetWin','Subliminal, attended, high prior: trial'); clf
    spm_MDP_VB_trial(MDP_2h{1},1:3,[3]);
    spm_figure('GetWin','Subliminal, attended, high prior: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_2h{1});
    spm_figure('GetWin','Subliminal, attended, high prior: LFP'); clf
    spm_MDP_VB_LFP(MDP_2h{1},[],1);
    
    spm_figure('GetWin','Supraliminal, unattended, low prior: trial'); clf
    spm_MDP_VB_trial(MDP_3l{1},1:3,[1,3:4]);
    spm_figure('GetWin','Supraliminal, unattended, low prior: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_3l{1});
    spm_figure('GetWin','Supraliminal, unattended, low prior: LFP'); clf
    spm_MDP_VB_LFP(MDP_3l{1},[],1);
    spm_figure('GetWin','Supraliminal, unattended, neutral: trial'); clf
    spm_MDP_VB_trial(MDP_3n{1},1:3,[1,3:4]);
    spm_figure('GetWin','Supraliminal, unattended, neutral: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_3n{1});
    spm_figure('GetWin','Supraliminal, unattended, neutral: LFP'); clf
    spm_MDP_VB_LFP(MDP_3n{1},[],1);
    spm_figure('GetWin','Supraliminal, unattended, high prior: trial'); clf
    spm_MDP_VB_trial(MDP_3h{1},1:3,[1,3:4]);
    spm_figure('GetWin','Supraliminal, unattended, high prior: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_3h{1});
    spm_figure('GetWin','Supraliminal, unattended, high prior: LFP'); clf
    spm_MDP_VB_LFP(MDP_3h{1},[],1);
    
    spm_figure('GetWin','Supraliminal, attended, low prior: trial'); clf
    spm_MDP_VB_trial(MDP_4l{1},1:3,[1,3:4]);
    spm_figure('GetWin','Supraliminal, attended, low prior: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_4l{1});
    spm_figure('GetWin','Supraliminal, attended, low prior: LFP'); clf
    spm_MDP_VB_LFP(MDP_4l{1},[],1);
    spm_figure('GetWin','Supraliminal, attended, neutral: trial'); clf
    spm_MDP_VB_trial(MDP_4n{1},1:3,[1,3:4]);
    spm_figure('GetWin','Supraliminal, attended, neutral: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_4n{1});
    spm_figure('GetWin','Supraliminal, attended, neutral: LFP'); clf
    spm_MDP_VB_LFP(MDP_4n{1},[],1);
    spm_figure('GetWin','Supraliminal, attended, high prior: trial'); clf
    spm_MDP_VB_trial(MDP_4h{1},1:3,[1,3:4]);
    spm_figure('GetWin','Supraliminal, attended, high prior: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_4h{1});
    spm_figure('GetWin','Supraliminal, attended, high prior: LFP'); clf
    spm_MDP_VB_LFP(MDP_4h{1},[],1);
    
    %% Plot ERPs
    
    %Extracted ERPs
    
    [u1l,v1l ,ind] = spm_MDP_VB_ERP_PGNW(MDP_1l{1},1);%subliminal unattended low prior
    [u1n,v1n ,ind] = spm_MDP_VB_ERP_PGNW(MDP_1n{1},1);%subliminal unattended neutral
    [u1h,v1h ,ind] = spm_MDP_VB_ERP_PGNW(MDP_1h{1},1);%subliminal unattended high prior
    
    [u2l,v2l ,ind] = spm_MDP_VB_ERP_PGNW(MDP_2l{1},1);%subliminal attended low prior
    [u2n,v2n ,ind] = spm_MDP_VB_ERP_PGNW(MDP_2n{1},1);%subliminal attended neutral
    [u2h,v2h ,ind] = spm_MDP_VB_ERP_PGNW(MDP_2h{1},1);%subliminal attended high prior
    
    [u3l,v3l ,ind] = spm_MDP_VB_ERP_PGNW(MDP_3l{1},1);%supraliminal unattended low prior
    [u3n,v3n ,ind] = spm_MDP_VB_ERP_PGNW(MDP_3n{1},1);%supraliminal unattended neutral
    [u3h,v3h ,ind] = spm_MDP_VB_ERP_PGNW(MDP_3h{1},1);%supraliminal unattended high prior
    
    [u4l,v4l ,ind] = spm_MDP_VB_ERP_PGNW(MDP_4l{1},1);%supraliminal attended low prior
    [u4n,v4n ,ind] = spm_MDP_VB_ERP_PGNW(MDP_4n{1},1);%supraliminal attended neutral
    [u4h,v4h ,ind] = spm_MDP_VB_ERP_PGNW(MDP_4h{1},1);%supraliminal attended high prior
    
    i   = cumsum(ind);
    i   = i(3) + (-64:-1);

    u1l  = u1l(i,:);
    v1l  = v1l(i,:);
    u1n  = u1n(i,:);
    v1n  = v1n(i,:);
    u1h  = u1h(i,:);
    v1h  = v1h(i,:);

    u2l  = u2l(i,:);
    v2l  = v2l(i,:);
    u2n  = u2n(i,:);
    v2n  = v2n(i,:);
    u2h  = u2h(i,:);
    v2h  = v2h(i,:);

    u3l  = u3l(i,:);
    v3l  = v3l(i,:);
    u3n  = u3n(i,:);
    v3n  = v3n(i,:);
    u3h  = u3h(i,:);
    v3h  = v3h(i,:);
    
    u4l  = u4l(i,:);
    v4l  = v4l(i,:);
    u4n  = u4n(i,:);
    v4n  = v4n(i,:);
    u4h  = u4h(i,:);
    v4h  = v4h(i,:);
    
    pst = (1:length(i))*1000/144;
    line(1:length(pst)) = 0;
   
    
    % --taxonomy
    
    %----- First level ERPs
    limits = [0 400 -1 1];
    
    spm_figure('GetWin','Taxonomy ERPs: First level'); clf

    subplot(3,2,1)
    hold on
    plot(pst,sum(v1n,2),'k', 'LineWidth',2) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    hold off
    
    subplot(3,2,2)
    hold on
    plot(pst,sum(v2n,2),'k', 'LineWidth',2) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    hold off
    
    subplot(3,2,3)
    hold on
    plot(pst,sum(v3n,2),'k', 'LineWidth',2) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    hold off
    
    subplot(3,2,4)
    hold on
    plot(pst,sum(v4n,2),'k', 'LineWidth',2) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    hold off

    %----- Second level ERPs
    limits = [0 400 -.2 .38];
    
    spm_figure('GetWin','Taxonomy ERPs: Second level'); clf
    
    
    subplot(3,2,1)
    hold on
    plot(pst,sum(u1n,2),'r', 'LineWidth',3) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    hold off
    
    subplot(3,2,2)
    hold on
    plot(pst,sum(u2n,2),'r', 'LineWidth',3) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    hold off
    
    subplot(3,2,3)
    hold on
    plot(pst,sum(u3n,2),'r', 'LineWidth',3) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    hold off
    
    subplot(3,2,4)
    hold on
    plot(pst,sum(u4n,2),'r', 'LineWidth',3) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    hold off
   
    
    % - Extended taxonomy
    
    limits = [250 400 -.2 .38];
    
    spm_figure('GetWin','ExTax'); clf
    
    subplot(3,2,1)
    plot(pst,sum(u4n,2) ,'r', 'LineWidth',1) 
    hold on 
    plot(pst,sum(u4h,2),'Color', [0 .5 0],'LineWidth',1) 
    plot(pst,sum(u4l,2),'b', 'LineWidth',1) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    title('P3b','Fontsize',16)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    
    subplot(3,2,2)
    plot(pst,sum(u3n,2) ,'r', 'LineWidth',2) 
    hold on 
    plot(pst,sum(u3h,2),'Color', [0 .5 0],'LineWidth',1) 
    plot(pst,sum(u3l,2),'b', 'LineWidth',1) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    title('P3b','Fontsize',16)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    
    subplot(3,2,3)
    plot(pst,sum(u2n,2) ,'r', 'LineWidth',2) 
    hold on 
    plot(pst,sum(u2h,2),'Color', [0 .5 0],'LineWidth',1) 
    plot(pst,sum(u2l,2),'b', 'LineWidth',1) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    title('P3b','Fontsize',16)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    
    subplot(3,2,4)
    plot(pst,sum(u1n,2) ,'r', 'LineWidth',2) 
    hold on 
    plot(pst,sum(u1h,2),'Color', [0 .5 0],'LineWidth',1) 
    plot(pst,sum(u1l,2),'b', 'LineWidth',1) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    title('P3b','Fontsize',16)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    
    % Sustained inattentional blindness
    %======================================================================

    spm_figure('GetWin','Phase 1: trial'); clf
    spm_MDP_VB_trial(MDP_P1{1},1:3);
    spm_figure('GetWin','Phase 1: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_P1{1});
    spm_figure('GetWin','Phase 1: LFP'); clf
    spm_MDP_VB_LFP(MDP_P1{1},[],1);

    spm_figure('GetWin','Phase 2: trial'); clf
    spm_MDP_VB_trial(MDP_P2{1},1:3);
    spm_figure('GetWin','Phase 2: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_P2{1});
    spm_figure('GetWin','Phase 2: LFP'); clf
    spm_MDP_VB_LFP(MDP_P2{1},[],1);

    spm_figure('GetWin','Phase 3: trial'); clf
    spm_MDP_VB_trial(MDP_P3{1},1:3);
    spm_figure('GetWin','Phase 3: ERP'); clf
    spm_MDP_VB_ERP_PGNW(MDP_P3{1});
    spm_figure('GetWin','Phase 3: LFP'); clf
    spm_MDP_VB_LFP(MDP_P3{1},[],1);

    %Extracted ERPs
    
    [u1,v1 ,ind] = spm_MDP_VB_ERP_PGNW(MDP_P1{1},1); %Phase 1
    [u2,v2 ,ind] = spm_MDP_VB_ERP_PGNW(MDP_P2{1},1); %Phase 2
    [u3,v3 ,ind] = spm_MDP_VB_ERP_PGNW(MDP_P3{1},1); %Phase 3

    i   = cumsum(ind);
    i   = i(3) + (-64:-1);

    u1  = u1(i,:);
    v1  = v1(i,:);

    u2  = u2(i,:);
    v2  = v2(i,:);

    u3  = u3(i,:);
    v3  = v3(i,:);

    pst = (1:length(i))*1000/144;
    
    limits = [0 400 -.4 .4];

    spm_figure('GetWin','SIB: ERPs'); clf

    subplot(3,2,1)
    hold on
    plot(pst,sum(u1,2) ,'r', 'LineWidth',3) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    hold off

    subplot(3,2,2)
    hold on
    plot(pst,sum(u2,2),'r','LineWidth',3) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    hold off

    subplot(3,2,3)
    hold on
    plot(pst,sum(u3,2),'r', 'LineWidth',3) 
    plot(pst,line,'k:', 'LineWidth',2) 
    xlabel('Peristimulus time (ms)'), ylabel('Depolarisation')
    axis(limits)
    ax = gca;
    ax.YDir = 'reverse';
    set(gca,'FontSize',20)
    hold off
    
end 
