import torch
from langchain_huggingface import HuggingFacePipeline
from openai import OpenAI
import time
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from transformers import pipeline
from constants import model_name, llm_analysis_model

template_bullet = """
 You are a helpful assistant that summarizes conversations.

 Summarize the following dialog. Identify the key points and exchanges between speakers.
 Use bullet points to describe important statements or shifts in topic.
 Preserve who said what when it's important and identify the speakers by name. Also, give a sentiment analysis for the
 conversation as it progresses.
 ```{text}```

 BULLET POINT SUMMARY:
 """
template_sentiment = """
 You are a helpful assistant that does sentiment analysis.

 identify the speakers by name. give a sentiment analysis for the
 conversation as it progresses.  Finally, give a sentiment SCORE for the overall 
 conversationon on a scale of -1 to 1, where -1 is extremely negative and 1 is extremely positive. label the sections as shown.

 ```{text}```

 SENTIMENT ANALYSIS:
 
 SENTIMENT SCORE:
 """


def use_openai(input, template):
    prompt = PromptTemplate(template=template, input_variables=["text"])

    llm = ChatOpenAI(
        # model="gpt-3.5-turbo-instruct",
        model="gpt-4.1",
        temperature=0.5,
        max_tokens=None,)

    chain = prompt | llm | StrOutputParser()

    out = chain.invoke(input)

    print(out)
    return out

def use_huggingface2(input, template):
    # Define the model name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        # max_length=1000,
        max_new_tokens=800,
        eos_token_id=tokenizer.eos_token_id
    )

    # Wrap in LangChain-compatible LLM
#     text ="""The Trump administration is taking a major step to super-charge deportation efforts by deputizing law enforcement across the country to carry out immigration enforcement, a Department of Homeland Security memo exclusively reviewed by The Daily Wire reveals.
# The memo lays out that DHS has signed 562 agreements with state and local law enforcement groups, taking advantage of a federal law that allows the agencies to enforce certain aspects of immigration law. While these sorts of agreements existed in the previous administration, the number of agreements signed since Trump took office is already a 316% increase from what was signed during four years of the Biden administration, the memo says.
# The agreements could be one of the biggest keys to boosting deportation numbers, which members of the administration have reportedly grown frustrated with. With an estimated 18.6 million illegal aliens currently present in the United States and a judiciary that has continuously stonewalled the Trump administration’s attempts to deport illegal aliens, the federal government may otherwise struggle to remove a significant share of the illegal population without support from law enforcement partners.
# Under the agreement, state and local law enforcement agencies are able to conduct immigration enforcement operations with ICE oversight, use federal immigration databases to identify the status of arrestees, and initiate ICE detainers to hold illegal aliens until ICE can gain custody of them, providing what could be a massive force multiplier in the Trump administration’s effort to deport at least one million illegal aliens per year.
# The administration says its collaboration with state and local law enforcement is yielding results.
# “Since January 20, our partnerships with state and local law enforcement have led to over 1,100 arrests of dangerous criminal illegal aliens,” the Homeland Security memo says. “ICE has also trained over 5,000 officers across the country to help track down and arrest criminal illegal aliens in their communities.”
# One far-left organization backed by George Soros’s Open Society Foundations complained that these agreements “are designed to extend the reach of the Trump deportation machine.” Now, they’re proving to be an effective force multiplier for the Trump administration as it seeks to enforce federal immigration law.
# One of these agreements resulted in Operation Tidal Wave, a first-of-its-kind joint immigration enforcement effort from Homeland Security and Florida."""

    llm = HuggingFacePipeline(
        pipeline=text_gen_pipeline,
        model_kwargs={
            "temperature": 0.5,  # Lower = more focused and factual
            "top_k": 50,  # Limits selection to top 50 tokens by probability
            "top_p": 0.85,  # Cumulative probability sampling
            "repetition_penalty": 1.2,  # Penalizes repeated phrases
            "max_new_tokens": 300  # Ensures full, rich output
        }
    )

    prompt = PromptTemplate(template=template, input_variables=["text"])
    chain = prompt | llm | StrOutputParser()

    out = chain.invoke(input)

    print(out)
    return out


def transcript_analysis(transcript):
    results = []
    input=""
    for speaker in transcript:
        input += speaker + "\n"

    start = time.time()
    if(llm_analysis_model == "openai"):
        response1 = use_openai(input,template_bullet)
        response2 = use_openai(input,template_sentiment)
    elif llm_analysis_model == "hugging_face":
        response1 = use_huggingface2(input, template_bullet) 
        response2 = use_huggingface2(input, template_sentiment) 

    stop = time.time()
    elapsed=stop-start
    print("transcript analysis consumed " + str(elapsed))

    transcript = response1[0:response1.index("BULLET POINT SUMMARY:")]
    summary = response1[response1.index("BULLET POINT SUMMARY:"):]
    sentiment = response2[response2.index("SENTIMENT ANALYSIS:"):response2.index("SENTIMENT SCORE:")]
    sentiment_score  = response2[response2.index("SENTIMENT SCORE:"):]
    
    return transcript, summary, sentiment, sentiment_score
