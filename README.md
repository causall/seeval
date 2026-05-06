# Seeval

### Overview
A simple tool to make generating synthetic datasets easier or at least more systematic and fast.

### Usage
pip install seevals
git clone https://github.com/causal/seevals.git


### Architecture

So take a jsonl log file  convert it into evaluation dataset file , to work along side the criteria defined for evaluating the log data


Once you have evaluation dataset transformed you will take this data and then you want to build a grader for it , it should be straight forward
but its not quiet because the data wasn't generated properly so you need to transform the data before hand so it's in a better format upfront and mirrors what people would have in reality. That will remove some of the awkward preprocessing step that's then computed. I should also pre group the cohorts so it's alot more understandable. 