# so the idea is that I want to be able to generate a model

# here's a list of models to generate data from here's the dspy module

# MyModule should take JSON extraction in mixed mode

#MyModule.forward(): TypeDataResponse. pydantic.DataClass

dataclass ConcreteClass:
    age: int
    name: str

dataclass ResponseData<TypeData>:
    responseData:<TypeData>
    filePath: Optional<str>

# inputs into the list
def a_list_generator<Typed>(data_list:List<Typed>, Module): generate
        return agenerator_func()

# needs to be thread safe
def agenerator_func(DSPMyModule):
    datalist = ["football", "basketball", "soccer"]
    return def generate():

def run()

# here we define a small helper llm prompt that will allow you to do input data and then output data to do a comparison into the output. It will also produce a file that you can export to then load
# it will also do sample population statistics for your data
def manual_grader(data, path, scale=False):
    return { filePath: path, }
exampleGenerator = agenerator_func(DSPMyModule)
# generate examples
noDisk = true
exampleGenerator = a_list_generator(["football","basketball"],MyModule)

responses:Response<List<ConcreteClass>> = generate(exampleGenerator, num_samples, noDisk, parallelsim)
# The process is small scale manual inspection of data does the prompt generally do the right thing or not if it looks good then proceed to deep data dive
# interactive_display(responses.responseData)

# here we generate a single shot html page that will allow you to yes or no score
manual_grader(reponses.responseData, scale=True, path="")

stats = sample_statistic()
print(stats)

# if stats look good then produce a larger value or generate more samples and grade those to get a better sample population statistic







manual_eval(responses)
