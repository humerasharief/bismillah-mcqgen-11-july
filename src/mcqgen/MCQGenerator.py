import os 
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.mcqgen.logger import logging

#importing necessary packages from langchain
from langchain.chat_models import ChatOpenAI
from langchain .llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_community.callbacks.manager import get_openai_callback

# Loading the environment variables from the .env file 
load_dotenv()

# Access the environment variables 
KEY = os.getenv('OPEN_API_KEY_1')

llm = ChatOpenAI(openai_api_key=KEY, model_name='gpt-3.5-turbo', temperature='0.5')



TEMPLATE="""
Text{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sute the questions are not repeated and check all the questions to be confirmed the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs
{response_json}

"""

quize_generation_prompt = PromptTemplate(
    input_variables=['text','number','subject','tone','response_json'],
    template=TEMPLATE
)

quiz_chain = LLMChain(llm=llm, prompt=quize_generation_prompt, output_key='quiz', verbose=True)


TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quize_evaluation_prompt = PromptTemplate(
    input_variables=['subject','quiz'], 
    template=TEMPLATE2
)

review_chain = LLMChain(llm=llm, prompt=quize_evaluation_prompt, output_key='review', verbose=True)


# Combining both quiz_chain and review_chain into a single chain  
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain], 
    input_variables=['text','number','subject','tone','response_json'],
    output_variables=['quiz','review'],
    verbose=True
)