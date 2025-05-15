from pprint import pprint

import torch
from langchain_huggingface import HuggingFacePipeline
import re
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
 Preserve who said what when it's important and identify the speakers by name.
 ```{text}```

 BULLET POINT SUMMARY:
 """
template_sentiment = """
 You are a helpful assistant that does sentiment analysis.

 identify the speakers by name. in the section labeled "SENTIMENT ANALYSIS" below, give a sentiment analysis for the
 conversation as it progresses.  
 
 Also, in the "SENTIMENT SCORE" section below give a sentiment SCORE for the overall 
 conversation as a number on a scale of -1 to 1, where -1 is extremely negative and 1 
 is extremely positive. Do not use punctuation for sentiment score. 
 
 label the sections as shown below.

 SENTIMENT ANALYSIS:
 






 SENTIMENT SCORE:



 ```{text}```
 """


def use_openai(input, template):
    prompt = PromptTemplate(template=template, input_variables=["text"])

    llm = ChatOpenAI(
        # model="o3-2025-04-16",
        model="gpt-4.1",
        temperature=0.2,
        top_p=.25,
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

    llm = HuggingFacePipeline(
        pipeline=text_gen_pipeline,
        model_kwargs={
            "temperature": 0.3,  # Lower = more focused and factual
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
    else:
        raise Exception("An llm for transcript analysis must be chosen.")

    stop = time.time()
    elapsed=stop-start
    print("transcript analysis consumed " + str(elapsed))

    pprint(response1)
    pprint(response2)

    transcript = input
    summary = response1[response1.index("BULLET POINT SUMMARY:"):]
    sentiment = response2[response2.index("SENTIMENT ANALYSIS:"):response2.index("SENTIMENT SCORE:")]
    sentiment_score  = re.sub("[^0-9\.-]", "", response2[response2.index("SENTIMENT SCORE:"):])
    if sentiment_score[len(sentiment_score)-1] == '.':
        sentiment_score = sentiment_score[0: len(sentiment_score)-1]
    
    return transcript, summary, sentiment, sentiment_score
