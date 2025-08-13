import pydantic
import json
from typing import TypedDict, Type, Tuple, Dict, Iterable, ParamSpec, TypeVar, Generic, List, Callable, Optional, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import dspy
import agents
import data_types as types
from agents import ScenarioArgs
from pdb import Pdb

T = TypeVar('T')
R = TypeVar('R', bound=pydantic.BaseModel)

lm = dspy.LM(
    model="openai/bedrock-sonnet-37",
    #model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
   # lm = dspy.LM('bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0')
    api_base="http://localhost:4000",
    api_key="noop",
)

class ForwardModule(Protocol[T, R]):
    def forward(self, params: T, /)-> R:
        ...

class ResponseData(pydantic.BaseModel, Generic[T]):
    data: List[Optional[T]]

def run_parallel(module: ForwardModule[T, R],
    args_list: List[T],
    lm: dspy.LM,
    concurrency: int)-> ResponseData[R]:
    def executor(args: T):
        with dspy.context(lm=lm):
            return module.forward(args)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(executor, args): i for i, args in enumerate(args_list)}
        results: List[R | None] = [None] * len(args_list)
        # thread-safe because they write to different areas of memory
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
    return ResponseData(data=results)

def get_values(values: List[R|None])->List[R]:
    return [value for value in values if value is not None]
scenarios = [
  ScenarioArgs({"scenario": "The three sisters Cherokee growth cycle — influenced by soil nitrogen, planting sequence timing, precipitation variability, pest pressure, intercrop spacing, and varietal selection."}),
  ScenarioArgs({"scenario": "Urban traffic congestion during school drop-off windows — driven by overlapping bell times, limited curb length, bus reliability, parent mode choice, crossing-guard staffing, and weather."}),
  ScenarioArgs({"scenario": "Seasonal flu vaccination uptake and community immunity — shaped by clinic distance, out-of-pocket cost, employer incentives, risk perception, prior side effects, and reminder systems."}),
  ScenarioArgs({"scenario": "Emergency department crowding and inpatient discharge delays — affected by boarding time, bed availability, lab turnaround, triage accuracy, staffing mix, and discharge coordination."}),
  ScenarioArgs({"scenario": "Housing affordability pressures and urban sprawl — influenced by zoning constraints, interest rates, construction costs, wage growth, transit access, and short-term rental prevalence."}),
  ScenarioArgs({"scenario": "Social media engagement and misinformation spread — driven by algorithmic amplification, fact-check latency, creator monetization, network homophily, report friction, and bot activity."}),
  ScenarioArgs({"scenario": "Drought intensity, groundwater extraction, and farm yields — shaped by precipitation deficits, irrigation efficiency, pumping costs, crop selection, soil moisture retention, and water rights enforcement."}),
  ScenarioArgs({"scenario": "Wildfire risk, forest fuel loads, and regional air quality — influenced by fuel accumulation, drought stress, wind events, suppression capacity, WUI development, and prescribed burn policy."}),
  ScenarioArgs({"scenario": "Fisheries quotas, bycatch, and illegal catch dynamics — affected by stock assessment accuracy, enforcement intensity, gear selectivity, market prices, fisher incentives, and port monitoring."}),
  ScenarioArgs({"scenario": "Electric vehicle adoption and public charging build-out — driven by charger density, electricity rates, model availability, upfront price, range anxiety, and rebate programs."}),
  ScenarioArgs({"scenario": "Renewable energy intermittency and grid stability reserves — shaped by solar/wind variability, storage capacity, demand response participation, transmission constraints, reserve margins, and forecasting accuracy."}),
  ScenarioArgs({"scenario": "Remote work adoption and downtown retail vitality — influenced by employer policies, commute times, office vacancy rates, broadband quality, worker preferences, and transit frequency."}),
  ScenarioArgs({"scenario": "Public transit reliability, ridership, and fare revenue — driven by on-time performance, headways, fare policy, network coverage, safety perception, and transfer friction."}),
  ScenarioArgs({"scenario": "Youth vaping prevalence and regulatory enforcement — affected by retail compliance checks, flavor availability, social supply networks, school enforcement, risk perception, and marketing exposure."}),
  ScenarioArgs({"scenario": "Obesity prevalence, food deserts, and physical activity — shaped by food prices, retail proximity, school lunch quality, park access, sedentary time, and cultural norms."}),
  ScenarioArgs({"scenario": "Opioid prescribing, treatment access, and overdose rates — influenced by guideline adherence, MAT availability, fentanyl prevalence, PDMP utilization, stigma, and naloxone distribution."}),
  ScenarioArgs({"scenario": "Antimicrobial resistance and antibiotic stewardship — driven by inappropriate prescribing, surveillance coverage, infection control, agricultural antibiotic use, rapid diagnostics adoption, and patient demand."}),
  ScenarioArgs({"scenario": "University enrollment, tuition pricing, and program quality — affected by demographics, financial aid generosity, labor market signaling, online program competition, campus amenities, and completion rates."}),
  ScenarioArgs({"scenario": "Small business cash flow, credit access, and hiring — shaped by invoice payment delays, interest rates, collateral requirements, demand seasonality, wage expectations, and regulatory burden."}),
  ScenarioArgs({"scenario": "Consumer electronics supply chain bullwhip effects — influenced by demand forecasting accuracy, lead times, component shortages, order batching, channel inventory visibility, and promotional cycles."}),
  ScenarioArgs({"scenario": "Hospital readmissions and post-discharge follow-up — driven by discharge instructions quality, primary care access, social determinants, medication reconciliation, home health availability, and telehealth use."}),
  ScenarioArgs({"scenario": "Urban tree canopy, heat islands, and electricity demand — affected by planting rates, species mix, maintenance budgets, impermeable surface area, building insulation, and heat-wave frequency."}),
  ScenarioArgs({"scenario": "Coastal erosion, seawalls, and beach tourism — shaped by storm frequency, sediment supply, sea-level rise, hardening versus soft solutions, insurance costs, and visitor perception."}),
  ScenarioArgs({"scenario": "K-12 attendance, truancy interventions, and achievement — influenced by transport reliability, chronic illness, family economic stress, school climate, attendance incentives, and tutoring access."}),
  ScenarioArgs({"scenario": "Municipal water pricing, leakage, and conservation — driven by tiered rates, metering accuracy, pipe age, seasonal demand, public education, and drought restrictions."}),
  ScenarioArgs({"scenario": "Cybersecurity breaches, disclosure policies, and investment — affected by attack surface, employee training, patch latency, regulatory penalties, incident response maturity, and cyber insurance."}),
  ScenarioArgs({"scenario": "Urban crime rates, community trust, and policing tactics — shaped by clearance rates, economic opportunity, hotspot patrols, community engagement, pretrial policies, and social services availability."}),
  ScenarioArgs({"scenario": "Agricultural pesticide use, pest resistance, and yields — influenced by application timing, active ingredient rotation, pest monitoring, IPM adoption, weather windows, and crop prices."}),
  ScenarioArgs({"scenario": "NGO fundraising cycles and program impact — driven by donor churn, campaign timing, overhead transparency, grant dependency, impact reporting cadence, and economic conditions."}),
  ScenarioArgs({"scenario": "LLM content moderation feedback and user satisfaction — affected by false positive rates, appeals latency, guidance clarity, reinforcement signals, creator incentives, and transparency."})
]

results = run_parallel(agents.InterviewGenerationModule(),scenarios, lm, 20)
rubrics = [
    agents.Rubric(ge=0.0,le=0.30,desc="Plan overview is consistent with the entities and relationships"),
    agents.Rubric(ge=0.0,le=0.25,desc="Relationships make logical sense with the scenario"),
    agents.Rubric(ge=0.0,le=0.25,desc="Entities described have logical types"),
    agents.Rubric(ge=0.0,le=0.10,desc="No entities that are not in a relationship"),
    agents.Rubric(ge=0.0,le=0.10,desc="No relationships that contain entities that are not described")
]
criteria =agents.Criteria(rubrics=rubrics,max_total_score=1)

Val = TypeVar('Val')
def make_grading_inputs(criteria:agents.Criteria, inputs:List[Val])->List[agents.GradingInput[Val]]:
    grading_inputs = map( lambda val: agents.GradingInput[Val](criteria=criteria, input=val),
        inputs)
    return list(grading_inputs)
LT = TypeVar('LT',bound=pydantic.BaseModel)

def loader(path:str, LoadingType: Type[LT])->List[LT]:
    results:List[LT] = []
    with open(path, 'r') as f:
        for line in f:
            result = json.loads(line)
            results.append(LoadingType.model_validate_json(result.get("item", {})))
    return results

#input_data=list(map(lambda val: val.analysis_plan, get_values(results.data)))
# grading_inputs = make_grading_inputs(criteria,input_data)

results = loader('./results.jsonl', agents.AnalysisPlanningResult)
input_data=list(map(lambda val: val.analysis_plan, results))

grades = run_parallel(agents.GraderGenerationModule(),grading_inputs, lm, 20)


# generate synthetic data
# validate synthetic data
# use synthetic data to train model
# validate trained model on expected results



with open("./results.jsonl", 'w') as f:
    for value in get_values(results.data):
        f.write(json.dumps({"item": value.analysis_plan.model_dump()}) + "\n")

import pdb
pdb.set_trace()
print(results)
