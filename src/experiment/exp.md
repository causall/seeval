So one thought here is that what I have is a system for aligning llms to individual preferences, one that I can test to verify that alignment
so when I turn a generic judge into a specific judge this judge is trying to do what the humans do.

In this case if the humans have a set of irrational preferences we're trying to learn those preferences more specifically.

We can then see how the artifical preferences align
with the machine generated preferences for arbitrary data. If the data is sufficiently varied
we can craft artifical datasets that encode this behavior.

This behavior then allows us to learn the underlying preference selection behaviors .

Is there a real data set like HelpSteer3 that I can use to encapsulate the preferences of the humans

The innovation here is that we're able to do this conversion without fine tuning a model, and without
knowing how to prompt the preferences apriori

In theory this setup would allow for more realtime extraction on adjustment of llm perf.

### NeurIPS 2025 best paper

https://openreview.net/pdf?id=saDOrrnNTz

### Question here can we improve this with our preference tuning ?

We aim to assess how LMs, reward models, and LM judges align with human ratings when evaluating
alternative responses to open-ended queries

### Code issue here

Motivation for comparing model scores to average human ratings. Our motivation stems from
how reward models (or LM judges) are used in training to evaluate responses to open-ended queries
without a single ground truth. Different annotators may prefer different answers, yet their average
ratings are often similar, implying multiple responses can be equally high-quality. Current reward
models, however, fail to capture this equivalence, assigning diverging scores and causing downstream
models to overvalue one response despite comparable human approval. To address this, we collect 25
human ratings per example to capture diverse preferences, using the average score to reflect shared
human judgment. We then test whether LMs, reward models, and LM judges correlate less reliably
with responses that humans broadly consider comparably good, hence our choice to compute human
correlation using average human ratings.

The paper that I just read was really amazing,
I think what stuck out to me was how clear and clean the metrics were the target was also very clean .

So what does it look like if I had data preference aligned generic prompt optimization. We move from uncorrelated responses to much more correlated responses that we can align across the test set of data . In specific domains

This means we can create judges that match human preferences more easily for feedback in downstream tasks?

Why is a judge as llm that matches human preferences important , and how do you construct one from raw data like how do you ask the right questions.

You have 5 attributes and different personas have
different weights on those attributes we want to move default preferences to mirror that of the persona, selected.

We want to also see what multi tenant preference
would look like if we could steer the outcome with prompt optimization alone in the average direction

Then the strict rating of the outomce according to those preferences to see if they can be steered into the average outcome of those preferences.

I'm trying to change the structure to meet my expectatiosn where I take a group and learn their preference , i"m trying to leverage underlying llm knowledge to aid in that preference taking..

So how do we do that efficiently
If I sample random users and have a mixture of rated movies by all parties and weights is that the way to do it to learn individual preferences or can I learn a group preference and is the initial approach more right

Where I find deltas on preference per genre , where it assumes genre is the right feature , to replicate scoring patterns for the average,

Human agent delegation made it easy to have humans to have an agent to have on your own bot

yep having your personal assistant and agent, human agent delegation is key

So it's where it can help in that process,

Governance for delegation voting, well you need preferences of the user , but you need to know what you stand for so the best playground is daos and crypto, so we might need a good way to delegate to people and to infer from preferences .

Crypto forward in governance / tally.so / conviction voting

you can signal everytime what you want , so like the protocol wants to raise taxes you'll vote no or once for future votes in that direction

If a vote like that comes up, and mirror your preference there.

Cyber punk user based on what type of voting,

Voting agent asks questions to vote for and the prompt represents you I think, so you know it's not exploitable at the end of the day what matters.

So can we have a continuous soft signal so like preferences, so where the questions are not exploitable

Do you want everyone , to want the ai to vote

claude , gets warnings about freedome of searching the internet and install repositories from anywhere or is there , so it's a bit scary to do it

Virtualization, so security risks are difficult and tricky

So who stores all these data that's important

IMPORT preferences from facebook, twitter, instagram, linkedin ?

IMPORT your data from chatgpt ?

Reality of you as observed by others -

clawdbot coding assistant ?

SuperNet is building privacy-enabled, portable AI context memory. It allows users to own, control, and monetize their personal context across AI applications, unlocking true personalization without lock-in. Through a GTM browser extension that acts as a secure data wallet, users can seamlessly collect, manage, and inject their context into any AI app while preserving privacy.SuperNet’s infrastructure fuses on-chain vector precompiles, compute, search, and storage with a novel data-gating primitive, turning personal context into a secure, portable, revenue-sharing digital asset. Its SDK enables Web3 wallets, Web2 apps, publishers, and adtech platforms to build new businesses around user-owned context data.

So preferences for voting, but business preference for a representation of hte preferences and the agents negotatiate the share price there is a back and forth a communication ,

So the information must be pulled vs pushed

So how do you have a preference guard dog ,

So personalized and randomized questions

Preference bot must be not front facing and must be siloed behind a proxy

System to represent preferences, for simulations

So we try and approximate if you put llms to represent a userbase, you might be able to use it to represent the agents in the behaviors, and test the products in simulations with the agentic representatiosn of the customers.

Ok so wonce we have the preference direction we know about how much more strongly they
feel about any one movie, but the group preference can be synthesized by approval rate

where approval rate is

ar = the preference direction normalized we will use tanh as a smoothing function on
the z score that we have the user preference direction which will bound our score between
[-1,1] this will then allow us to then set a preference threshold in conjunction with

so then you can predict how the group approves

2 = tanh(c/z)

so in our case we will be choosing tanh(z/c) where c is 1.5 or 1.0 as the two segments we'll scale by where we have clear approval
