endTime = 0.005;

tau = 0.1;
tau1 = 0.1;

k = 1;
r = 2;
l = 20;

m = 100;
p = 10;

alpha = 1.0;
beta = 1.0;

main_event(endTime,l,k,r,tau,tau1,m,p,alpha,beta)


%Softmax function, takes in a tau parameter and a normalized list of values for actions (should be the length of 'actions'); here these are estimated success probabilities:

function y = softmax (tau, values, actions)
divVals = values / tau;
expVals = exp (divVals);
total = sum(expVals);

for i = 1:length(actions)
    x = rand;
    if x < (expVals(i)/total)
        y = actions(i);
        break
    else 
        total = total - expVals(i);
    end   
end
end


%Define a process for making observations. Observe a random action and its utility (0 or 1).

function y = observe (rewardProbs, actions) %function name changed

obs = randi([0,length(actions)-1]);
x = rand; 
rew = x < rewardProbs(obs+1); 
y = [obs, rew]; %returning python-like indeces bc other functions handle this
end 


% The randomizing agent:

function values = randomizer(k, values,observation,rewardProbs,actions)

row = observation(1) + 1; %indexing starts at 1 \in matlab
col = observation(2) + 1;


values(row,col) = values(row,col) +1;

alt = randsample(actions,k);

for a = 1:length(alt)
    x = rand;
    draw = (x < rewardProbs(a)) + 1;
    values(a,draw) = values(a,draw) + 1; 
end

end



%The agent who maximizes information gain. In the entropy and expInfo functions,p is the number of successes so far and n is the number of failures, both for a particular action a. So the estimated probability of success for a is p/(p+n).
%PLEASE NOTE name change from 'entropy' to 'ntropy' 

function y = ntropy (p,n)
prob = p/(p+n);
y = -((prob*log2(prob)) + ((1-prob)*log2(1-prob)));
end

function y = expInfo (p,n)
y = ((p/(p+n)) * (ntropy(p,n) - ntropy(p+1,n)) + (n/(p+n)) * (ntropy(p,n) - ntropy(p,n+1)));
end


function y = info(k,values,observation,rewardProbs,actions)
values(observation(1)+1, observation(2)+1) = values(observation(1)+1, observation(2)+1) + 1;

for i = 1:k
    
    [rows,cols] = size(values);
    sums = zeros(1,rows);
    
    for row = 1:rows
        sums(row) = expInfo(values(row,2),values(row,1));
    end
    
    [M,a] = max(sums(:));
    
    x = rand;
    draw = (x < rewardProbs(a)) + 1;
    
    values(a,draw) = values(a,draw) + 1;
end

y = values;
end


%The agent who focuses on good actions. First a helper function that makes a pair [p,n] into a probability p/(p+n). [The name 'normalize' is an artifact of an earlier version.]

function result = normalize (values)
[rows,cols] = size(values);
result = zeros(1,rows);

for row = 1:rows
    result(row) = values(row,2)/(values(row,1)+values(row,2));   
end 
end


function y = good (k,tau,values,observation,rewardProbs,actions)
values(observation(1)+1, observation(2)+1) = values(observation(1)+1, observation(2)+1) + 1;

for i = 1:k
    %add 1 to a and draw because of different indexing in matlab
    a = softmax (tau, normalize(values), actions) + 1; 
    x = rand;
    draw = (x < rewardProbs(a)) + 1;
    values(a,draw) = values(a,draw) + 1; 
end

y = values;

end


% The agent who focuses on bad actions:

function y = bad (k,tau,values,observation,rewardProbs,actions)
values(observation(1)+1, observation(2)+1) = values(observation(1)+1, observation(2)+1) + 1;


for i = 1:k
    expRev = 1-normalize(values); 
    a = softmax(tau,expRev,actions) + 1;
    x = rand;
    draw = (x < rewardProbs(a)) + 1;
    values(a,draw) = values(a,draw) + 1;  
end

y = values;

end


% Learning proceeds for n trials, initializing each agent with [1,1] estimates for each action and producing a vector of estimated success probabilities for each agent. 

function y = learning (k,tau,rewardProbs,n,actions,unif)
    randValues = unif; %matlab copy behavior creates new
    infoValues = unif;
    goodValues = unif;
    badValues = unif;
 
    for i = 1:n
        observ = observe(rewardProbs,actions);
        randValues = randomizer(k,randValues,observ,rewardProbs,actions);
        infoValues = info(k,infoValues,observ,rewardProbs,actions);
        goodValues = good(k,tau,goodValues,observ,rewardProbs,actions);
        badValues = bad(k,tau,badValues,observ,rewardProbs,actions);
    end
    y = [normalize(randValues); normalize(infoValues); normalize(goodValues); normalize(badValues)];
end 


% This function is applied to each agent with their own vals (estimated probabilities), and  rewardProbs are the objective success probabilities for the actions. Agents softmax select r actions and then chose the one of these with the highest estimated value. [The way it is implemented, we keep a copy of the agent's reward probabilities, tempVals, where we simply set to 0 the reward probabilities for any action we've already selected.]

function y = actionSelection (r,tau1,vals,rewardProbs,actions)
    
    tempVals = vals;
    focus = [];
    
    for i = 1:r
        add = softmax(tau1,tempVals,actions);
        focus (1,i) = add;
        tempVals(1,add+1) = 0.0;
    end 
    probs = rewardProbs(focus+1);
    [M,I] = max(probs(:));

    y = focus(1,I);
end


%This is the test phase. We run m test trials. We could abbreviate this by simply calculating the agent's expected utility at this stage. That is, there will be a simple analytic expression for expected utility and we could just calculate this given the rewardProbs, the estimated reward probabilities and the softmax parameter tau1. In this case, the actionSelection function above would also not be needed. The following might be easier though for larger values of r.

function y = test(k,r,tau,tau1,m,rewardProbs,n,actions,unif)

learn = learning(k,tau,rewardProbs,n,actions,unif);

randAgent = learn(1,:);
randSum = 0.0;

infoAgent = learn(2,:);
infoSum = 0.0;

goodAgent = learn(3,:);
goodSum = 0.0;

badAgent = learn(4,:);
badSum = 0.0;

for i = 1:m
    randNum = rand;
    
    randSum = randSum + (randNum < rewardProbs(1,actionSelection(r,tau1,randAgent,rewardProbs,actions)+1));
    infoSum = infoSum + (randNum < rewardProbs(1,actionSelection(r,tau1,infoAgent,rewardProbs,actions)+1));
    goodSum = goodSum + (randNum < rewardProbs(1,actionSelection(r,tau1,goodAgent,rewardProbs,actions)+1));
    badSum = badSum + (randNum < rewardProbs(1,actionSelection(r,tau1,badAgent,rewardProbs,actions)+1));
end 

y = [(randSum/m),(infoSum/m),(goodSum/m),(badSum/m)];

end


% This defines a trial where the reward probabilities are drawn from Beta(alpha,beta)

function y = trial(k,r,tau,tau1,m,p,a,n,alpha,beta)
 actions = 0:(a-1);
 
 unif = ones(a,2); %array of doubles
 
 rS = 0.0;
 iS = 0.0;
 gS = 0.0;
 bS = 0.0;
 
 for i = 1:p
     rewardProbs = [];
     for i = 1:a
         rewardProbs(1,i) = betarnd(1.0,1.0);
     end 
     t = test(k,r,tau,tau1,m,rewardProbs,n,actions,unif);
     
     rS = rS + t(1,1);
     iS = iS + t(1,2);
     gS = gS + t(1,3);
     bS = bS + t(1,4);
 end 
 y = [(rS/p),(iS/p),(gS/p),(bS/p)];
end


% This averages over trials with 5, 10, 15, up to l.

function y = trials(l,k,r,tau,tau1,m,p,a,alpha,beta)

sumR = 0.0;
sumI = 0.0;
sumG = 0.0;
sumB = 0.0;

for i = 5:5:l
    t = trial(k,r,tau,tau1,m,p,a,i,alpha,beta);
    sumR = sumR + t(1,1);
    sumI = sumI + t(1,2);
    sumG = sumG + t(1,3);
    sumB = sumB + t(1,4);
end
sum = l/5;
y = [(sumR/sum),(sumI/sum),(sumG/sum),(sumB/sum)];
end

function y = main_event(endTime,l,k,r,tau,tau1,m,p,alpha,beta)

file = fopen('result.txt','w');

overallSumInfo = 0.0;
overallSumRand = 0.0;
overallSumGood = 0.0;
overallSumBad = 0.0;

overallNum = 0.0;  %number of trials overall

%This will be the best each agent has done so far over the second best in that condition. By default all 0.

bestInfo = [0.0,0.0,0.0,0.0];
bestRand = [0.0,0.0,0.0,0.0];
bestGood = [0.0,0.0,0.0,0.0];
bestBad = [0.0,0.0,0.0,0.0];

%We will start with 5 actions, and then increase until time is up:

startTime = now;
elapsed = 0.0;

a = 5;


while elapsed < endTime
    for alpha = 20:20:100
        alpha = double(alpha)/100;
        for beta = 20:20:100
            beta = double(beta)/100;
            res = trials(l,k,r,tau,tau1,m,p,a,alpha,beta);
            fprintf(file,'[%f, %f, %f] : [%f, %f, %f, %f]\n',a,alpha,beta,res(1,1),res(1,2),res(1,3),res(1,4));
            
            if (res(1,1) - max([res(1,2),res(1,3),res(1,4)])) > bestRand(1,1)
                bestRand = [(res(1,1) - max([res(1,2),res(1,3),res(1,4)])),a,alpha,beta];
            end

            if (res(1,2) - max([res(1,1),res(1,3),res(1,4)])) > bestInfo(1,1)
                bestInfo = [(res(1,2) - max([res(1,1),res(1,3),res(1,4)])),a,alpha,beta];
            end

            if (res(1,3) - max([res(1,1),res(1,2),res(1,4)])) > bestGood(1,1)
                bestGood = [(res(1,3) - max([res(1,1),res(1,2),res(1,4)])),a,alpha,beta];
            end 

            if (res(1,4) - max([res(1,1),res(1,2),res(1,3)])) > bestBad(1,1)
                bestBad = [(res(1,4) - max([res(1,1),res(1,2),res(1,3)])),a,alpha,beta];
            end 

            overallSumRand = overallSumRand + res(1,1);
            overallSumInfo = overallSumInfo + res(1,2);
            overallSumGood = overallSumGood + res(1,3);
            overallSumBad = overallSumBad + res(1,4);

            overallNum = overallNum + 1;

            disp(res)    
        end
    end
    elapsed = now - startTime;
    a = a + 5;
end 


fprintf(file,'\n');

fprintf(file,'The average for the random agent overall was %f\n',(overallSumRand/overallNum));
fprintf(file,'The average for the random agent overall was advantage of %f, with %f actions, alpha=%f, beta=%f (This will be 0 if it never did better)\n',bestRand(1,1),bestRand(1,2),bestRand(1,3),bestRand(1,4));

fprintf(file,'\n');

fprintf(file,'The average for the infomax agent overall was %f\n',(overallSumInfo/overallNum));
fprintf(file,'The average for the infomax agent overall was advantage of %f, with %f actions, alpha=%f, beta=%f\n',bestInfo(1,1),bestInfo(1,2),bestInfo(1,3),bestInfo(1,4));

fprintf(file,'\n');

fprintf(file,'The average for the good counterfactual sampling agent overall was %f\n',(overallSumGood/overallNum));
fprintf(file,'The average for the good counterfactual sampling agent overall was advantage of %f, with %f actions, alpha=%f, beta=%f\n',bestGood(1,1),bestGood(1,2),bestGood(1,3),bestGood(1,4));

fprintf(file,'\n');

fprintf(file,'The average for the bad counterfactual sampling agent overall was %f\n',(overallSumBad/overallNum));
fprintf(file,'The average for the bad counterfactual sampling agent overall was advantage of %f, with %f actions, alpha=%f, beta=%f\n',bestBad(1,1),bestBad(1,2),bestBad(1,3),bestBad(1,4));

end
