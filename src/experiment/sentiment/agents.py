import dspy
from experiment.sentiment.data_types import (
    SentimentGenerationArgs,
    SentimentHeadlineOutput,
)


class SentimentHeadlineGenerator(dspy.Signature):
    """
    You are an headline generator, you generate a list of headlines that are associated with a sentiment.
    """

    sentiment_args: SentimentGenerationArgs = dspy.InputField(
        description="the arguments for the sentiment headline generator"
    )
    headlines: SentimentHeadlineOutput = dspy.OutputField(
        description="the list of headlines that are associated with the sentiment"
    )


class SentimentHeadlineGeneratorModule(dspy.Module):
    def __init__(self):
        self.generator = dspy.ChainOfThought(SentimentHeadlineGenerator)

    def forward(self, args: SentimentGenerationArgs) -> dspy.Prediction:
        return self.generator(sentiment_args=args)

    def get_value(self, prediction: dspy.Prediction) -> SentimentHeadlineOutput:
        return prediction.headlines
