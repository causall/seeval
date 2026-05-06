# This experiment tests wethere or not it's possible to generically prompt optimize an llm from typed data with simplistic prompts driven by rubrics. In addition to this typed data, can the preferences or qualities of the data evaluations then drive net beneficial behaviors for groups of individuals?

If user A has knowledge of categories 1-10 and user B has knowledge of categories 10-20 , and user C has knowledge of categories 20-30 , can the information generated in aggregate with various weightings train or tune an llm to act better than their individual data parts.

Additionally can we re/weight the data of each individual according to some heuristic that is set.

The goal being community knowledge, to be utilized to prompt optimize llms in realish time without fine tuning ,

The ultimate being to solicit information from individuals that can lead to an understanding of a task, that can lead to a quality evaluator that is capable of being used in downstream tasks effectively

The downstream task is the real goal.

---

31questions how would it work

Let's say you wanted to write a title that would appeal to a specific demographic and you wanted to answer questions that lead to that

There's then a way to do attributed prompt analysis where we're also able to attribute the influence of subsets of data that contributed to the response

so collaborative data or community data can be signaled

So what I need to do is actually ru nthrough each user and compute the ratings per genre, then take the difference between that rating and their average rating and use that as a measure of bias in genre per rating per user, then compute this for the group of 100 randomly selected, then use that as a difference against the gen population along side the standard deviation of the gen pop genre computation.

This will then give me a way of saying that the group is statisically different in any of the genres, which implies, that within the result there should be something the grader learns that mirrors that same preference
in the genre, or is able to flat out compute the actual data sample.

