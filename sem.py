# from langchain.agents import load_tools
# from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.tools import BaseTool, StructuredTool, tool
from crewai import Agent, Task, Crew, Process
from crewai.tasks.task_output import TaskOutput
from langchain_openai import ChatOpenAI
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import CohereEncoder, OpenAIEncoder
# from crewai_tools import BaseTool, tool
import re
import os
import asyncio
import requests
from requests.exceptions import RequestException
import chainlit as cl
from chainlit import run_sync
from dotenv import load_dotenv

from pydantic import BaseModel, ValidationError, field_validator
from typing import Union
from datetime import datetime, date

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings

from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from pydantic import BaseModel
from typing import List, Dict, Any

from langchain_groq import ChatGroq




load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Chatbot - Deployed"

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
guard_llm = ChatGroq(temperature=0, model_name="llama-guard-3-8b")

embeddings = OpenAIEmbeddings()
agent_wise_memory = ConversationBufferMemory(return_messages=True)
chat_wise_memory = ConversationBufferMemory(return_messages=True) 

make_payment = Route(
    name="make_payment",
    utterances=[
        "I want to make a payment",
        "pay a bill",
        "Do a transaction",
        "move money",
        "pay credit card",
    ],
)

check_account_balance = Route(
    name="check_account_balance",
    utterances=[
        "I want to check my account balance",
        "my credit balance",
        "how much more can I spend",
        "What is my bank balance",
    ],
)

apply_credit_card = Route(
    name="apply_credit_card",
    utterances=[
        "Apply for a credit card",
        "Need a new card",
        "open a new credit card account",
        "Get a credit card",
    ],
)

dispute_help = Route(
    name="dispute_help",
    utterances=[
        "Dispute Help",
        "I need help with a dispute on my card or account",
        "assist me with a transaction dispute",
        "I want to contest a fee on my bill",
        "Isssue with a transaction",
        "Unauthorized transaction",
        "Create a new Dispute",
        "Check Dispute Status",
        "Update status of a dispute",
    ],
)

fallback_intent = Route(
    name="fallback_intent",
    utterances=[
        "Fallback Intent",
        "I don't understand",
        "I didn't get that",
        "I don't know",
        "something else",
        "Intent: 'Fallback Intent'"
    ],
)

routesMain = [make_payment, check_account_balance, apply_credit_card, dispute_help, fallback_intent]
encoder = OpenAIEncoder(name='text-embedding-3-small')
routerLayerMain = RouteLayer(encoder=encoder, routes=routesMain, llm=llm, top_k=3)


create_dispute = Route(
    name="create_dispute",
    utterances=[
        "Create a Dispute",
        "Create New Dispute",
        "Issue with a transaction",
        "unauthorized transaction",
    ],
)

check_dispute_status = Route(
    name="check_dispute_status",
    utterances=[
        "I want to check the status of a dispute",
        "What happened to the dispute I raised before",
        "Need an update on a dispute",
        "what's going on with the dispute",
    ],
)

update_dispute_status = Route(
    name="update_dispute_status",
    utterances=[
        "Update the status of a Dispute",
        "Give more information on a dispute",
    ],
)

fallback_dispute_type = Route(
    name="fallback_dispute_type",
    utterances=[
        "Anything else under disputes not related to creating a new dispute, checking the status of a dispute of updating a dispute",
    ],
)

routeSubDispute = [create_dispute, check_dispute_status, update_dispute_status]
routerLayerSubDispute = RouteLayer(encoder=encoder, routes=routeSubDispute, llm=llm, top_k=3)
    
    
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
  format_D="\n\n".join([d.page_content for d in docs])
  return format_D
    
@tool("Ask Question using the RAG")
def ask_rag_question(question: str) -> str:
    """Ask a question using the RAG model"""
    docsearch = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = docsearch.as_retriever()
    
    template = """Answer the question based only on the following context:
              {context}
              If you don't know the answer, just say out of scope, don't try to make up an answer.
              Question: {question}
              """
              
    prompt=ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_without_guardrails = rag_chain
    
    response = rag_chain_without_guardrails.invoke(question)
    return response


@tool("Ask Human follow up questions")
def ask_human(question: str) -> str:
    """Ask human follow up questions"""
    agent_wise_memory.chat_memory.add_ai_message(question)
    chat_wise_memory.chat_memory.add_ai_message(question)
    human_response = run_sync(cl.AskUserMessage(content=f"{question}", timeout=600).send())
    agent_wise_memory.chat_memory.add_user_message(human_response["output"])
    chat_wise_memory.chat_memory.add_user_message(human_response["output"])
    return human_response["output"]
  
      

@tool("Get Employee Details")
def check_account_balance_tool(user_id: str) -> str:
    """Get employee details"""
    url = f"https://6508f0e856db83a34d9cc3fa.mockapi.io/api/v1/getEmployee/BankUsers/{user_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        return f"Error: Unable to fetch data. Exception: {str(e)}"
    except ValueError:
        return "Error: Unable to decode the response."

class DisputeDetails(BaseModel):
    accountNumber: str
    merchantName: str
    status: str
    dateAndTime: date
    amount: float
    description: str
    
    @field_validator('dateAndTime', mode='before')
    def validate_date(cls, value):
        try:
            return date.fromisoformat(value)
        except ValueError as e:
            raise ValueError('Date must be in YYYY-MM-DD format')
    
    @field_validator('accountNumber')
    def validate_account_number(cls, value):
        if not re.match(r'^\d{4}-\d{4}-\d{4}$', value):
            raise ValueError('Account number must be in the format XXXX-XXXX-XXXX')
        return value
      
@tool("Create Dispute")
def create_dispute_tool(dispute_details: dict) -> str:
    """Create a new dispute"""
    
    try:
        validated_details = DisputeDetails(**dispute_details)
    except ValidationError as e:
        return f"Error: JSON body is not valid format. {e.json()}"
      
    url = "https://6508f0e856db83a34d9cc3fa.mockapi.io/api/v1/getEmployee/transaction"
    print(url);
    print(validated_details.model_dump(mode="json"));
    try:
        payload = validated_details.model_dump(mode="json")
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        return f"Error: Unable to create dispute. Exception: {str(e)}"
    except ValueError:
        return "Error: Unable to decode the response."
      
@tool("Get Dispute Details")
def get_dispute_tool(dispute_id: str) -> str:
    """Get dispute details"""
    url = f"https://6508f0e856db83a34d9cc3fa.mockapi.io/api/v1/getEmployee/transaction/{dispute_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        return f"Error: Unable to fetch dispute details. Exception: {str(e)}"
    except ValueError:
        return "Error: Unable to decode the response."
      

@tool("Update Dispute Status")
def update_dispute_status_tool(dispute_id: str, status: str) -> str:
    """Update the status of a dispute"""
    url = f"https://6508f0e856db83a34d9cc3fa.mockapi.io/api/v1/getEmployee/transaction/{dispute_id}"
    payload = {"status": status}
    try:
        response = requests.patch(url, json=payload)
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        return f"Error: Unable to update dispute status. Exception: {str(e)}"
    except ValueError:
        return "Error: Unable to decode the response."

@tool("Send Messages")
def send_messages_tool(message: str) -> str:
    """Send messages to the user"""
    run_sync(cl.Message(content=str(message)).send())




user_details = {
    "firstName": "Chathusha",
    "lastName": "Wijenayake",
    "fullName": "Chathusha Wijenayake",
    "email": "chathusha@gmail.com",
    "id": "78965432",
    "customerType": "Cards",
    "billPayees":[
       {
          "payeeName":"Just Energy",
       },
       {
          "payeeName":"AT&T",
       }
    ],
    "creditCardAccounts": [
        {
            "accountId": "1234-4567-8907",
            "accountName": "Credit Card...8907",
            "creditLimit": "10000",
            "availableCreditLimit": "9500",
            "totalBalance": "500",
            "statementDate": "06/20/2024",
            "minimumPayment": "50"
        },
        {
            "accountId": "9087-9876-7864",
            "accountName": "Delta Card...7864",
            "creditLimit": "8500",
            "availableCreditLimit": "6000",
            "totalBalance": "2500",
            "statementDate": "06/20/2024",
            "minimumPayment": "100"
        }
    ]
}


transactions = {
  "transactions": [
    {
      "merchant_name": "Adidas",
      "amount": 129.99,
      "date": "09-July-2024",
      "account_number": "1234567890"
    },
    {
      "merchant_name": "Ticketmaster",
      "amount": 54.65,
      "date": "08-July-2024",
      "account_number": "0987654321"
    },
    {
      "merchant_name": "Amazon",
      "amount": 23.50,
      "date": "08-July-2024",
      "account_number": "1122334455"
    },
    {
      "merchant_name": "Walmart",
      "amount": 8.75,
      "date": "07-July-2024",
      "account_number": "5566778899"
    },
    {
      "merchant_name": "Best Buy",
      "amount": 45.00,
      "date": "07-July-2024",
      "account_number": "2233445566"
    }
  ]
}

from trulens.dashboard import run_dashboard 
from trulens.core import Feedback
from trulens.core import TruSession
from trulens.apps.langchain import TruChain
from trulens.providers.openai import OpenAI as fOpenAI
from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider import OpenAI
from langchain.prompts import PromptTemplate
from typing import Tuple, Dict

session = TruSession()




# Initial Prompt + Conversation Feedback
# Initial Prompt + Conversation Feedback

class prompt_with_conversation_relevence(fOpenAI):
    def prompt_with_conversation_relevence_feedback(self, question: str) -> Tuple[float, Dict]:
        history = agent_wise_memory.load_memory_variables({})['history']
        
        formatted_history = ""
        for message in history:
            if message.__class__.__name__ == "AIMessage":
                formatted_history += "AI Message: " + message.content + "\n"
            elif message.__class__.__name__ == "HumanMessage":
                formatted_history += "Human Message: " + message.content + "\n"
        
        
        
        system_prompt = """You are a RELEVANCE grader; providing the relevance of the given CHAT_GUIDANCE to the given CHAT_CONVERSATION.
            Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

            A few additional scoring guidelines:

            - Long CHAT_CONVERSATION should score equally well as short CHAT_CONVERSATION.

            - RELEVANCE score should increase as the CHAT_CONVERSATION provides more RELEVANT context to the CHAT_GUIDANCE.

            - RELEVANCE score should increase as the CHAT_CONVERSATION provides RELEVANT context to more parts of the CHAT_GUIDANCE.

            - CHAT_CONVERSATION that is RELEVANT to some of the CHAT_GUIDANCE should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

            - CHAT_CONVERSATION that is RELEVANT to most of the CHAT_GUIDANCE should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

            - CHAT_CONVERSATION that is RELEVANT to the entire CHAT_GUIDANCE should get a score of 9 or 10. Higher score indicates more RELEVANCE.

            - CHAT_CONVERSATION must be relevant and helpful for answering the entire CHAT_GUIDANCE to get a score of 10.

            - Never elaborate."""
        
        user_prompt = PromptTemplate.from_template(
            """CHAT_GUIDANCE: {question}

            CHAT_CONVERSATION: {formatted_history}
            
            RELEVANCE: """
        )
        
        user_prompt = user_prompt.format(question = question, formatted_history = formatted_history)
        user_prompt = user_prompt.replace(
            "RELEVANCE:", prompts.COT_REASONS_TEMPLATE
        )
        
        result = self.generate_score_and_reasons(system_prompt, user_prompt)
        
        print("\n\n=====================================")
        print("Start Initial Prompt + Converssation Feedback")
        print("=====================================")
        
        result = result
        details = result[1]
        reason = details['reason'].split('\n')
        criteria = reason[0].split(': ')[1]
        print("Criteria:", criteria)
        supporting_evidence = reason[1].split(': ')[1]
        print("Supporting Evidence:", supporting_evidence)
        score = reason[-1].split(': ')[1]
        print("Score:", score)
        
        print("=====================================")
        print("End Initial Prompt + Converssation Feedback")
        print("=====================================\n\n")
        
        return result


prompt_with_conversation_relevence_custom = prompt_with_conversation_relevence()

f_prompt_with_conversation_relevence = Feedback(prompt_with_conversation_relevence_custom.prompt_with_conversation_relevence_feedback, verbose=True).on_input()

# Initial Prompt + Conversation Feedback
# Initial Prompt + Conversation Feedback


# Conversation + Final Answer Feedback
# Conversation + Final Answer Feedback

class conversation_with_answer_relevence(fOpenAI):
    def conversation_with_answer_relevence_feedback(self, answer: str) -> Tuple[float, Dict]:
        history = agent_wise_memory.load_memory_variables({})['history']
        
        formatted_history = ""
        for message in history:
            if message.__class__.__name__ == "AIMessage":
                formatted_history += "AI Message: " + message.content + "\n"
            elif message.__class__.__name__ == "HumanMessage":
                formatted_history += "Human Message: " + message.content + "\n"
        
        
        
        system_prompt = """You are a RELEVANCE grader; providing the relevance of the given CHAT_GUIDANCE to the given CHAT_CONVERSATION.
            Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

            A few additional scoring guidelines:

            - Long FINAL_ANSWER should score equally well as short FINAL_ANSWER.

            - RELEVANCE score should increase as the FINAL_ANSWER provides more RELEVANT context to the CHAT_CONVERSATION.

            - RELEVANCE score should increase as the FINAL_ANSWER provides RELEVANT context to more parts of the CHAT_CONVERSATION.

            - FINAL_ANSWER that is RELEVANT to some of the CHAT_CONVERSATION should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

            - FINAL_ANSWER that is RELEVANT to most of the CHAT_CONVERSATION should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

            - FINAL_ANSWER that is RELEVANT to the entire CHAT_CONVERSATION should get a score of 9 or 10. Higher score indicates more RELEVANCE.

            - FINAL_ANSWER must be relevant and helpful for answering the entire CHAT_CONVERSATION to get a score of 10.

            - Never elaborate."""
        
        user_prompt = PromptTemplate.from_template(
            """
            FINAL_ANSWER: {answer}

            CHAT_CONVERSATION: {formatted_history}
            
            RELEVANCE: """
        )
        
        user_prompt = user_prompt.format(answer = answer, formatted_history = formatted_history)
        user_prompt = user_prompt.replace(
            "RELEVANCE:", prompts.COT_REASONS_TEMPLATE
        )
        
        result = self.generate_score_and_reasons(system_prompt, user_prompt)
        
        print("\n\n=====================================")
        print("Start Conversation + Final Answer Feedback")
        print("=====================================")
        
        result = result
        details = result[1]
        reason = details['reason'].split('\n')
        criteria = reason[0].split(': ')[1]
        print("Criteria:", criteria)
        supporting_evidence = reason[1].split(': ')[1]
        print("Supporting Evidence:", supporting_evidence)
        score = reason[-1].split(': ')[1]
        print("Score:", score)
        
        print("=====================================")
        print("End Conversation + Final Answer Feedback")
        print("=====================================\n\n")
        
        return result


conversation_with_answer_relevence_custom = conversation_with_answer_relevence()

f_conversation_with_answer_relevence = Feedback(conversation_with_answer_relevence_custom.conversation_with_answer_relevence_feedback, verbose=True).on_output()

# Conversation + Final Answer Feedback
# Conversation + Final Answer Feedback


# Initial Prompt + Converssation + Final Answer Feedback
# Initial Prompt + Converssation + Final Answer Feedback

class promt_with_conversation_and_answer_relevence(fOpenAI):
    def promt_with_conversation_and_answer_relevence_feedback(self, question: str, answer: str) -> Tuple[float, Dict]:
        history = agent_wise_memory.load_memory_variables({})['history']
        
        formatted_history = ""
        for message in history:
            if message.__class__.__name__ == "AIMessage":
                formatted_history += "AI Message: " + message.content + "\n"
            elif message.__class__.__name__ == "HumanMessage":
                formatted_history += "Human Message: " + message.content + "\n"
        
        
        
        system_prompt = """You are a RELEVANCE grader; providing the relevance of the given CHAT_GUIDANCE to the given CHAT_CONVERSATION.
            Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

            A few additional scoring guidelines:

            - Long CHAT_CONVERSATION should score equally well as short CHAT_CONVERSATION.
            
            - Long FINAL_ANSWER should score equally well as short FINAL_ANSWER.

            - RELEVANCE score should increase as the CHAT_CONVERSATION and FINAL_ANSWER provides more RELEVANT context to the CHAT_GUIDANCE.

            - RELEVANCE score should increase as the CHAT_CONVERSATION and FINAL_ANSWER provides RELEVANT context to more parts of the CHAT_GUIDANCE.

            - CHAT_CONVERSATION and FINAL_ANSWER that is RELEVANT to some of the CHAT_GUIDANCE should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

            - CHAT_CONVERSATION and FINAL_ANSWER that is RELEVANT to most of the CHAT_GUIDANCE should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

            - CHAT_CONVERSATION and FINAL_ANSWER that is RELEVANT to the entire CHAT_GUIDANCE should get a score of 9 or 10. Higher score indicates more RELEVANCE.

            - CHAT_CONVERSATION and FINAL_ANSWER must be relevant and helpful for answering the entire CHAT_GUIDANCE to get a score of 10.

            - Never elaborate."""
        
        user_prompt = PromptTemplate.from_template(
            """
            CHAT_GUIDANCE: {question}
            
            FINAL_ANSWER: {answer}

            CHAT_CONVERSATION: {formatted_history}
            
            RELEVANCE: """
        )
        
        user_prompt = user_prompt.format(question = question, answer = answer, formatted_history = formatted_history)
        user_prompt = user_prompt.replace(
            "RELEVANCE:", prompts.COT_REASONS_TEMPLATE
        )
        
        result = self.generate_score_and_reasons(system_prompt, user_prompt)
        
        print("\n\n=====================================")
        print("Start Initial Prompt + Converssation + Final Answer Feedback")
        print("=====================================")
        
        result = result
        details = result[1]
        reason = details['reason'].split('\n')
        criteria = reason[0].split(': ')[1]
        print("Criteria:", criteria)
        supporting_evidence = reason[1].split(': ')[1]
        print("Supporting Evidence:", supporting_evidence)
        score = reason[-1].split(': ')[1]
        print("Score:", score)
        
        print("=====================================")
        print("End Initial Prompt + Converssation + Final Answer Feedback")
        print("=====================================\n\n")
        
        return result


promt_with_conversation_and_answer_relevence_custom = promt_with_conversation_and_answer_relevence()

f_promt_with_conversation_and_answer_relevence = Feedback(promt_with_conversation_and_answer_relevence_custom.promt_with_conversation_and_answer_relevence_feedback, verbose=True).on_input_output()


# Initial Prompt + Converssation + Final Answer Feedback
# Initial Prompt + Converssation + Final Answer Feedback


# Initial Prompt + Final Answer Feedback
# Initial Prompt + Final Answer Feedback

class prompt_with_final_answer_relavence(fOpenAI):
    def prompt_with_final_answer_relavence_feedback(self, question: str, answer: str) -> Tuple[float, Dict]:
        
        
        
        system_prompt = """You are a RELEVANCE grader; providing the relevance of the given CHAT_GUIDANCE to the given QUESTION.
            Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

            A few additional scoring guidelines:

            - Long FINAL_ANSWER should score equally well as short FINAL_ANSWER.

            - RELEVANCE score should increase as the FINAL_ANSWER provides more RELEVANT context to the QUESTION.

            - RELEVANCE score should increase as the FINAL_ANSWER provides RELEVANT context to more parts of the QUESTION.

            - FINAL_ANSWER that is RELEVANT to some of the QUESTION should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

            - FINAL_ANSWER that is RELEVANT to most of the QUESTION should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

            - FINAL_ANSWER that is RELEVANT to the entire QUESTION should get a score of 9 or 10. Higher score indicates more RELEVANCE.

            - FINAL_ANSWER must be relevant and helpful for answering the entire QUESTION to get a score of 10.

            - Never elaborate."""
        
        user_prompt = PromptTemplate.from_template(
            """
            FINAL_ANSWER: {answer}

            QUESTION: {question}
            
            RELEVANCE: """
        )
        
        user_prompt = user_prompt.format(answer = answer, question = question)
        user_prompt = user_prompt.replace(
            "RELEVANCE:", prompts.COT_REASONS_TEMPLATE
        )
        
        result = self.generate_score_and_reasons(system_prompt, user_prompt)
        
        print("\n\n=====================================")
        print("Initial Prompt + Final Answer Feedback")
        print("=====================================")
        
        result = result
        details = result[1]
        reason = details['reason'].split('\n')
        criteria = reason[0].split(': ')[1]
        print("Criteria:", criteria)
        supporting_evidence = reason[1].split(': ')[1]
        print("Supporting Evidence:", supporting_evidence)
        score = reason[-1].split(': ')[1]
        print("Score:", score)
        
        print("=====================================")
        print("Initial Prompt + Final Answer Feedback")
        print("=====================================\n\n")
        
        return result


prompt_with_final_answer_relavence_custom = prompt_with_final_answer_relavence()

f_prompt_with_final_answer_relavence = Feedback(prompt_with_final_answer_relavence_custom.prompt_with_final_answer_relavence_feedback, verbose=True).on_input_output()   

# Initial Prompt + Final Answer Feedback
# Initial Prompt + Final Answer Feedback


harmless_feedbacks = [
    f_prompt_with_final_answer_relavence,
    f_prompt_with_conversation_relevence,
    f_conversation_with_answer_relevence,
    f_promt_with_conversation_and_answer_relevence,
]

def update_crew(agent : Agent, task: Task):
    items_crew.agents.pop(0)
    items_crew.agents.append(agent)
    items_crew.tasks.pop(0)
    items_crew.tasks.append(task)
    



  

def callback_function(output: TaskOutput):
    print("Output: ", output)
    query = str(output)
    print(query)

    if not query.startswith("Intent:") and not query.startswith("Utterance:"):
      run_sync(cl.Message(content=output).send())

    match = re.search(r"Intent: '(.+)', Utterance: '(.+)', Type: '(.+)'", query)
    
    # Intent: 'Dispute a transaction', Utterance: 'There is a transaction in my account and seems like I didn't do that', Type: 'MAIN'
    inputs = {
        'user_details': user_details,
        'chat_history': agent_wise_memory
    }

    if match:
      intent = match.group(1)
      utterance = match.group(2)
      taskType = match.group(3)
      print("Intent: ", intent)
      print("Utterance: ", utterance)
      print("Type: ", taskType)
      if taskType == "MAIN":
        if intent == "Fallback Intent":
            inputs['utterance'] = utterance
            inputs['intent'] = "Fallback Intent"
            update_crew(rag_agent, rag_agent_task)
        elif intent == "FAQ Intent":  
            inputs['utterance'] = utterance
            inputs['intent'] = "FAQ Intent"
            update_crew(fallback_agent, fallback_task)
        else:
          inputs['utterance'] = utterance
          if intent == "NOT FOUND":
            route = routerLayerMain(utterance)
            inputs['intent'] = route.name
            print("Route Name: ", route.name)
            if route.name == "make_payment":
              update_crew(make_payment_agent, make_payment_task)
            elif route.name == "check_account_balance":
              update_crew(check_account_balance_agent, check_account_balance_task)
            elif route.name == "apply_credit_card":
              update_crew(apply_credit_card_agent, apply_credit_card_task)
            elif route.name == "dispute_help":
              update_crew(dispute_help_agent, dispute_help_task)
            elif route.name == "fallback_intent":
              update_crew(fallback_agent, fallback_task)
            else:
              update_crew(fallback_agent, fallback_task)
              
          else:
            route = routerLayerMain(intent)
            print("Route Name: ", route.name)
            inputs['intent'] = route.name
            if route.name == "make_payment":
              update_crew(make_payment_agent, make_payment_task)
            elif route.name == "check_account_balance":
              update_crew(check_account_balance_agent, check_account_balance_task)
            elif route.name == "apply_credit_card":
              update_crew(apply_credit_card_agent, apply_credit_card_task)
            elif route.name == "dispute_help":
              update_crew(dispute_help_agent, dispute_help_task)
            elif route.name == "fallback_intent":
              update_crew(fallback_agent, fallback_task)
            else:
              update_crew(fallback_agent, fallback_task)
      elif taskType == "SUB":
        route = routerLayerMain(intent)
        print("Route Name Sub: ", route.name)
        inputs['intent'] = route.name
        inputs['utterance'] = utterance
        if route.name == "dispute_help":
          routeSubCheckDispute = routerLayerSubDispute(intent)
          print("Route Name Selected Sub: ", routeSubCheckDispute.name)
          
          if intent == "Fallback Dispute Type":
            update_crew(dispute_help_agent, dispute_help_sub_faq_task)
          else:
            if routeSubCheckDispute.name == "create_dispute":
              inputs['transactions_list'] = transactions
              update_crew(dispute_help_agent, dispute_help_sub_create_dispute)
            elif routeSubCheckDispute.name == "check_dispute_status":
              update_crew(dispute_help_agent, dispute_help_sub_check_dispute_status)
            elif routeSubCheckDispute.name == "update_dispute_status":
              update_crew(dispute_help_agent, dispute_help_sub_update_dispute_status)
            else:
              update_crew(dispute_help_agent, dispute_help_sub_faq_task)
              
    else:
      update_crew(intermediate_intent_recognizer, intermediate_recognize_the_intent)
      
    tru_agent = create_tru_agent(items_crew)
    agent_wise_memory.clear()
    # print_feedback_output()
    with tru_agent as recording:
      items_crew.kickoff(inputs=inputs)

    
initial_intent_recognizer = Agent(
    role='Initial Intent Recognizer',
    goal=f'Identify intention of the customer in Banking context',
    verbose=True,
    llm=llm,
    backstory="""Identify the customer intention and assign the task to the suitable banking agent""",
    cache=False,
    memory=False
)

initial_recognize_the_intent = Task(
  description=f"""
Please use following guidlines and identify the customer intention.
    
1. **Engage with the Customer:**
   Greet the customer warmly by their name always and mention that you are Alex, the banking assistant. Then, ask them clearly and kindly what assistance they need. Use a friendly and conversational tone to ensure they feel comfortable sharing their needs.
  Consider the user's details here: {{user_details}}
   
2. **Identify Intent:**
   From the customer's input, determine the most appropriate intent from the following list:
    
    - Intent 1 - Make Payment
    - Description 1: Help the customer make a payment or pay a bill.
    - Utterances 1: I want to make a payment, pay a bill, do a transaction, move money, pay credit card.

    - Intent 2 - Check Account Balance
    - Description 2: Help the customer check their account balance.
    - Utterances 2: I want to check my account balance, my credit balance, how much more can I spend, what is my bank balance.

    - Intent 3 - Apply for a Credit Card
    - Description 3: Help the customer apply for a credit card.
    - Utterances 3: Apply for a credit card, need a new card, open a new credit card account, get a credit card.

    - Intent 4 - Dispute Help
    - Description 4: Help the customer with a dispute on their card or account.
    - Utterances 4: Dispute help, I need help with a dispute on my card or account, assist me with a transaction dispute, I want to contest a fee on my bill, issue with a 
    transaction, unauthorized transaction, create a new dispute, check dispute status, update status of a dispute.
    
    - Intent 5 - Transfer Money
    - Description 5: Help the customer transfer money between accounts.
    - Utterances 5: Transfer money, move funds, send money to another account, transfer to savings.

    - Intent 6 - Open a New Account
    - Description 6: Help the customer open a new bank account.
    - Utterances 6: Open a new account, start a savings account, create a checking account, new bank account.

    - Intent 7 - Close an Account
    - Description 7: Help the customer close an existing bank account.
    - Utterances 7: Close my account, terminate my account, cancel my account, delete my account.

    - Intent 8 - Loan Inquiry
    - Description 8: Provide information on available loans and how to apply.
    - Utterances 8: Loan options, apply for a loan, loan rates, how to get a loan, personal loan information.

    - Intent 9 - Report Lost or Stolen Card
    - Description 9: Assist the customer in reporting a lost or stolen card.
    - Utterances 9: Report lost card, my card is stolen, lost my credit card, report stolen debit card.

    - Intent 10 - Update Personal Information
    - Description 10: Help the customer update their personal information.
    - Utterances 10: Update my information, change my address, update my phone number, change my email.

    - Intent 11 - Check Transaction History
    - Description 11: Help the customer check their transaction history.
    - Utterances 11: Check my transaction history, view my past transactions, transaction record, account activity.

    - Intent 12 - Set Up Direct Deposit
    - Description 12: Assist the customer in setting up direct deposit.
    - Utterances 12: Set up direct deposit, direct deposit information, enroll in direct deposit, direct deposit form.

    - Intent 13 - Reset Password
    - Description 13: Help the customer reset their online banking password.
    - Utterances 13: Reset my password, forgot my password, change my online banking password, can't log in.

    - Intent 14 - Find ATM/Branch
    - Description 14: Help the customer find the nearest ATM or branch location.
    - Utterances 14: Find an ATM, nearest branch, locate an ATM, where is the closest bank branch.

    - Intent 15 - Update Account Settings
    - Description 15: Help the customer update account settings such as notification preferences.
    - Utterances 15: Update my account settings, change notification preferences, update security settings, change account settings.

    - Intent 16 - Request Overdraft Protection
    - Description 16: Help the customer request overdraft protection for their account.
    - Utterances 16: Request overdraft protection, set up overdraft protection, overdraft services, overdraft coverage.

    - Intent 17 - Mobile Banking Assistance
    - Description 17: Help the customer with issues related to mobile banking.
    - Utterances 17: Mobile banking help, app not working, issues with mobile app, can't log into mobile banking.

    - Intent 18 - Account Statements
    - Description 18: Help the customer get their account statements.
    - Utterances 18: Get my account statement, view my statements, download bank statements, monthly statement.

    - Intent 19 - Request a Checkbook
    - Description 19: Help the customer request a new checkbook.
    - Utterances 19: Request a checkbook, order new checks, need more checks, checkbook request.

    - Intent 20 - Fallback - Intent
    - Description 20: If the customer's - intention is none of the above.
    - Utterances 20: I need help, I don't understand, can you assist me?
    
    
3. **Instructions**
    Consider Intent, Description, and Utterances to identify the correct intent. According to Description and Utterances, select the most appropriate intent from the given list. If the customer's query does not match any of the intents, select "Fallback Intent." 

4. **Clarification for Multiple Intents:**
   If the customer's request seems to contain to multiple intents, list the potential intents back to them and ask for clarification to confirm the correct one.

5. **Handling Descriptive Queries:**
   Should the customer ask a more detailed question, analyze their request step by step to identify the correct intent from the given list, without repeatedly asking the same question.

6. **Fallback Intent:**
   If you cannot determine one clear intent from the above list, select "Fallback Intent."

7. **Asking Further Questions:**
   Use the "ask_human" to generate questions for the customer, ensuring the questions are straightforward and conducive to identifying the correct intent.

8. **Goal:**
   Aim to identify the correct intent on the first attempt and avoid repeating questions unnecessarily. Always strive to provide a single, clear intent.


  """,
  expected_output="""
  Don't give JSON format. I need output in following format. Nothing else:
  If Identified Intent is equal to 'Fallback Intent': 
  Example : Intent: 'Fallback Intent', Utterance: 'User query that input by customer', Type: 'MAIN'

  If Identified Intent other than 'Fallback Intent' intent: 
  Example : Intent: 'Intent name', Utterance: 'User query that input by customer', Type: 'MAIN'
  """,
  agent=initial_intent_recognizer,
  tools=[ask_human],
  verbose=True,
  callback=callback_function,
)

intermediate_intent_recognizer = Agent(
    role='Intermediate Intent Recognizer',
    goal=f'Identify banking customer intention and assign customer to the respective banking agent',
    verbose=True,
    llm=llm,
    backstory="""You need to identify the customer need and you need to assign the task to the suitable banking agent""",
    cache=False,
    memory=False
)

intermediate_recognize_the_intent = Task(
  description=f"""
    Please use following guidlines and identify the customer intention.
    
1. **Engage with the Customer:**
   You need to help the customer after he completes his previous task.
  Therefore, you have to ask the customer if he needs more help from you in a human-friendly and convincing manner.
  Consider the user's details here: {{user_details}}
   
2. **Identify Intent:**
   From the customer's input, determine the most appropriate intent from the following list:
    - Intent 1 - Make Payment
    - Description 1: Help the customer make a payment or pay a bill.
    - Utterances 1: I want to make a payment, pay a bill, do a transaction, move money, pay credit card.

    - Intent 2 - Check Account Balance
    - Description 2: Help the customer check their account balance.
    - Utterances 2: I want to check my account balance, my credit balance, how much more can I spend, what is my bank balance.

    - Intent 3 - Apply for a Credit Card
    - Description 3: Help the customer apply for a credit card.
    - Utterances 3: Apply for a credit card, need a new card, open a new credit card account, get a credit card.

    - Intent 4 - Dispute Help
    - Description 4: Help the customer with a dispute on their card or account.
    - Utterances 4: Dispute help, I need help with a dispute on my card or account, assist me with a transaction dispute, I want to contest a fee on my bill, issue with a 
    transaction, unauthorized transaction, create a new dispute, check dispute status, update status of a dispute.
    
    - Intent 5 - Transfer Money
    - Description 5: Help the customer transfer money between accounts.
    - Utterances 5: Transfer money, move funds, send money to another account, transfer to savings.

    - Intent 6 - Open a New Account
    - Description 6: Help the customer open a new bank account.
    - Utterances 6: Open a new account, start a savings account, create a checking account, new bank account.

    - Intent 7 - Close an Account
    - Description 7: Help the customer close an existing bank account.
    - Utterances 7: Close my account, terminate my account, cancel my account, delete my account.

    - Intent 8 - Loan Inquiry
    - Description 8: Provide information on available loans and how to apply.
    - Utterances 8: Loan options, apply for a loan, loan rates, how to get a loan, personal loan information.

    - Intent 9 - Report Lost or Stolen Card
    - Description 9: Assist the customer in reporting a lost or stolen card.
    - Utterances 9: Report lost card, my card is stolen, lost my credit card, report stolen debit card.

    - Intent 10 - Update Personal Information
    - Description 10: Help the customer update their personal information.
    - Utterances 10: Update my information, change my address, update my phone number, change my email.

    - Intent 11 - Check Transaction History
    - Description 11: Help the customer check their transaction history.
    - Utterances 11: Check my transaction history, view my past transactions, transaction record, account activity.

    - Intent 12 - Set Up Direct Deposit
    - Description 12: Assist the customer in setting up direct deposit.
    - Utterances 12: Set up direct deposit, direct deposit information, enroll in direct deposit, direct deposit form.

    - Intent 13 - Reset Password
    - Description 13: Help the customer reset their online banking password.
    - Utterances 13: Reset my password, forgot my password, change my online banking password, can't log in.

    - Intent 14 - Find ATM/Branch
    - Description 14: Help the customer find the nearest ATM or branch location.
    - Utterances 14: Find an ATM, nearest branch, locate an ATM, where is the closest bank branch.

    - Intent 15 - Update Account Settings
    - Description 15: Help the customer update account settings such as notification preferences.
    - Utterances 15: Update my account settings, change notification preferences, update security settings, change account settings.

    - Intent 16 - Request Overdraft Protection
    - Description 16: Help the customer request overdraft protection for their account.
    - Utterances 16: Request overdraft protection, set up overdraft protection, overdraft services, overdraft coverage.

    - Intent 17 - Mobile Banking Assistance
    - Description 17: Help the customer with issues related to mobile banking.
    - Utterances 17: Mobile banking help, app not working, issues with mobile app, can't log into mobile banking.

    - Intent 18 - Account Statements
    - Description 18: Help the customer get their account statements.
    - Utterances 18: Get my account statement, view my statements, download bank statements, monthly statement.

    - Intent 19 - Request a Checkbook
    - Description 19: Help the customer request a new checkbook.
    - Utterances 19: Request a checkbook, order new checks, need more checks, checkbook request.

    - Intent 20 - Fallback - Intent
    - Description 20: If the customer's - intention is none of the above.
    - Utterances 20: I need help, I don't understand, can you assist me?
    
    
3. **Instructions**
    Consider Intent, Description, and Utterances to identify the correct intent. According to Description and Utterances, select the most appropriate intent from the given list. If the customer's query does not match any of the intents, select "Fallback Intent." 
    After customer give the answer, don't ask questions again and again. You need to identify the correct intent from the given list.

4. **Clarification for Multiple Intents:**
   If the customer's request seems to contain to multiple intents, list the potential intents back to them and ask for clarification to confirm the correct one.

5. **Handling Descriptive Queries:**
   Should the customer ask a more detailed question, analyze their request step by step to identify the correct intent from the given list, without repeatedly asking the same question.

6. **Fallback Intent:**
   If you cannot determine one clear intent from the above list, select "Fallback Intent."

7. **Asking Further Questions:**
   Use the "ask_human" to generate questions for the customer, ensuring the questions are straightforward and conducive to identifying the correct intent.

8. **Goal:**
   Aim to identify the correct intent on the first attempt and avoid repeating questions unnecessarily. Always strive to provide a single, clear intent.
  """,
  expected_output="""
  Don't give JSON format. I need output in following format. Nothing else:
  If Identified Intent is equal to 'Fallback Intent': 
  Example : Intent: 'Fallback Intent', Utterance: 'User query that input by customer', Type: 'MAIN'

  If Identified Intent other than 'Fallback Intent' intent:
  Example : Intent: 'Intent name', Utterance: 'User query that input by customer', Type: 'MAIN'
  """,
  agent=intermediate_intent_recognizer,
  tools=[ask_human],
  verbose=True,
  callback=callback_function,
)

make_payment_agent = Agent(
  role='Banking Assistant',
  goal='Request details from the user to make a payment or pay a bill',
  backstory="""You are a banking assistant and you need to collect details from the banking customer and help to make a payment or pay a bill""",
  verbose=True,
  llm=llm,
  cache=False,
  memory=False
)

make_payment_task = Task(
  description=f"""

***Objective***
Think you as a banking assistant. You need to extract essential information from the "user_details" represented as JSON, such as name, email, customer type, and id. You will offer assistance to the user on making a payment based on this data.

***Instructions***
If customer has multiple banking/card accounts, prompt them to choose an account by displaying the last four digits of each account ID. Once an account is selected, get the account ID, account name, credit limit, available credit limit, total balance, statement date, and minimum payment linked to that specific account for task execution.

Don't use greetings like "Hello" or "Hi" every time. Greet the user by their name when appropreate to establish the convincing conversation.

Consider the user's details here: {{user_details}}

Pay attention to the information mentioned in the user's query - {{utterance}}. If any specifications related to the payment and bill pay are included, utilize that and avoid repetitively asking for the same information.

You need to gather the following details: 
   - Account Number (mandatory): Extract from user_details. If multiple accounts, ask the user to select by showing the last 4 digits of each account ID. If user enter invalid or wrong account number, ask the user to provide the correct account number.
   - Bill Payee (mandatory): Can be extracted from the user's details under billPayees. Show all bill payees to the user and ask to select one. 
   - Pay amount (mandatory): Request this from the user. Confirm the details if provided in the user's input query, and use it for the task upon confirmation.
   - Date need to process the payment (mandatory): Request a date to process the payment.

Approach the user's query sequentially and try to identify the correct "Bill Payee", "Pay amount" and "Date need to process the payment" without redundantly asking the customer. If you manage to extract any details from the user's query, request individual confirmation from the user, and don't ask all the details at once.

Verify the "Pay amount" against the "creditLimit" and "availableCreditLimit" of the chosen account. If the transfer amount surpasses the available credit limit, prompt the user to enter a valid transfer amount.
   
When inquiring the first question, explain the task to the user and make sure all necessary inputs are provided. Continue the query until all mandatory details are collected. 

Use the "ask_human" tool for questioning. 

***Additional guidelines***
Ask the user again only if the reply or answer to any question is confusing and a decision cannot be made from the reply. 

If the user interrupts the task and provide irrelevant information to this task, confirm with them politely whether to terminate the current task or proceed. If the user consents to interrupt the current task, conclude the task execution.
    
""",
  expected_output="""
  Don't give JSON format. give the output in following format. Nothing else:
  If execute task successfully: Show proper human-friendly and convincing message to the user.
  
  If user enter any irrelavent reply and end the task output :  irrelavent utterance that customer entered to the flow with Intent: 'NOT FOUND' and Utterance: 'User's query' and Type: 'MAIN' keywords
  Example : Intent: 'NOT FOUND', Utterance: 'User query that input by customer', Type: 'MAIN'
  """,
  agent=make_payment_agent,
  tools=[ask_human],
  verbose=True,
  callback=callback_function,
)


check_account_balance_agent = Agent(
  role='Banker',
  goal='get user id and help to check account balance to the user',
  backstory="""You are a banking assistant and you need to collect details from the banking coustomer and help to them to check account balance""",
  verbose=True,
  llm=llm,
  cache=False,
  memory=False
)

check_account_balance_task = Task(
  description=f"""

    ### Objective
    Think you as a customer support banker. Your task is to help a user to check the account balance. Gather essential details from the user following these instructions:

    ### User Information
    Details are provided in JSON format within the "User Details" section, including name, email, customer type, and ID. If there are multiple bank or card accounts, display the last four digits of each account ID and ask the user to select one. Use the selected account's details for further processing.

    ### Instructions for Interaction
    - Do not greet the user with "Hello," "Hi," etc. If possible, use the user's name.
    - User Details: {{user_details}}
    - User's Input Query: {{utterance}}

    ### Data Collection Requirements
      -**Account Number (mandatory)**
      - Extract from user details.
      - If multiple accounts, ask the user to select by showing the last 4 digits of each account ID.

    ### Interaction Flow
    - Begin by clearly explaining the task to the user.
    - For each required detail, if mentioned in the user's query, confirm it. If not, ask for it.
    - If user provide more than one details once, gather them all and confirm.
    - Do not ask for all details at once; gather and confirm information step-by-step.
    - After the user enters the account number, according to the account number you can recognize the available credit limit in the user's selected account.

    ### Tools
    - **ask_human**: Use to ask the user any questions.

    ### Additional Guidelines
    - Ensure all mandatory details are collected.
    - If any information is missing, continue asking until all requirements are met. Ask the user again only if the reply or answer to any question is confusing and a decision cannot be made from the reply.
    - If the user interrupts the task and provide irrelevant information to this task, confirm with them politely whether to terminate the current task or proceed. If the user consents to interrupt the current task, conclude the task execution.
  """,
  expected_output="""
Please provide the output in the following format based on the scenario:

1. If the task is executed successfully:
   
   Display available credit limit with selected account number and a human-friendly message to the user.

2. If the user enters irrelevant content and the task ends:

   `Intent: 'NOT FOUND', Utterance: 'CUSTOMER_QUERY', Type: 'MAIN'`
   
   **Example:** `Intent: 'NOT FOUND', Utterance: 'User query that input by customer', Type: 'MAIN'`

Ensure the output matches exactly the specified formats above depending on whether the task is executed successfully or if the content is irrelevant.
  """,
  agent=check_account_balance_agent,
  tools=[ask_human],
  verbose=True,
  callback=callback_function,
)



dispute_help_agent = Agent(
  role='Banker',
  goal='Request required details to fulfill the customer need on disputes',
  backstory="""You are a banking assistant and you need to collect details from the banking customer about the dispute and help them complete the task on disputes""",
  verbose=True,
  llm=llm,
  cache=False,
  memory=False
)

dispute_help_task = Task(
  description=f"""

  **Objective**
    Think you as a customer support banker. Your task is to collect necessary details from the user to assist them with their dispute related query.



**Instructions:**
1. User's Input Query: {{utterance}}
2. If the user's input query already includes the necessary details, use those details to identify the dispute type without asking for additional information.
3. There are four types of intents under disputes:
    1. Intent 1: Initiate a Dispute
      Description: Help the customer initiate a financial dispute.
      Utterances: Initiate a dispute, I want to dispute a transaction, I need to contest a charge, I want to dispute a transaction, I want to dispute a charge, I want to dispute a fee, I want to dispute a payment
      
    2. Intent 2: Check the Status of a Dispute
      Description: Help the customer check the status of a dispute.
      Utterances: Check the status of a dispute, What happened to the dispute I raised before, Need an update on a dispute, what's going on with the dispute
      
    3. Intent 3: Update the Status of a Dispute
      Description: Help the customer update the status of a dispute.
      Utterances: Update the status of a dispute, How can I update the status of a dispute
      
    4. Intent 4: Fallback Dispute Type
      Description: If the customer's query does not match any of the intents above, set it as "Fallback Dispute Type" without displaying this type to the user.

**Instructions**
Consider Intent, Description, and Utterances to identify the correct intent. According to Description and Utterances, select the most appropriate intent from the given list. If the customer's query does not match any of the intents, select "Fallback Intent." 
          
**Process:**
1. Analyze the user's input query.
2. Identify and extract relevant details related to the dispute.
3. Determine the appropriate dispute intent based on the user's query.
4. If the query type does not match "Initiate a Dispute" "Check the Status of a Dispute" or "Update the Status of a Dispute" set it as "Fallback Dispute Type" without displaying this type to the user.


**Additional Guidelines:**
1. If the customer asks a descriptive question or provides incomplete information, think step-by-step to identify the dispute type without repeatedly asking the user for the same information.
2. When initiating a question to the user, clearly mention the task at hand.
3. Ensure all mandatory details are provided. If any mandatory input is missing, continue asking the user until all required information is obtained using the "ask_human" tool.
4. If the user interrupts the task and provide irrelevant information to this task, confirm with them politely whether to terminate the current task or proceed. If the user consents to interrupt the current task, conclude the task execution.
""",
  expected_output="""
  Please provide the output in the following format based on the scenario:

1. If the task is executed successfully:
   
    Intent: 'DISPUTE_TYPE', Utterance: 'CUSTOMER_QUERY', Type: 'SUB'
   
    Example: Intent: 'Selected Intent', Utterance: 'User query that input by customer', Type: 'SUB'

2. If the user enters irrelevant content and the task ends:

    Intent: 'NOT FOUND', Utterance: 'CUSTOMER_QUERY', Type: 'MAIN'
   
    Example: Intent: 'NOT FOUND', Utterance: 'User query that input by customer', Type: 'MAIN'

Ensure the output matches exactly the specified formats above depending on whether the task is executed successfully or if the content is irrelevant. 
  """,
  agent=dispute_help_agent,
  callback=callback_function,
)


dispute_help_sub_create_dispute = Task(
  description=f"""  

### Objective    
You are a banking assistant. Your task is to help a banking user to initiate a financial dispute. Gather essential details from the user following these instructions:

### User Information
Details are provided in JSON format within the "User Details" section, including name, email, customer type, and ID. If there are multiple bank or card accounts, display the last four digits of each account ID and ask the user to select one. Use the selected account's details for further processing.
 
### Instructions for Interaction
- Do not greet the user with "Hello," "Hi," etc. If possible, use the user's name.
- User Details: {{user_details}}
- User's Input Query: {{utterance}}
- Recent Transactions: {{transactions_list}}

### Data Collection Requirements
1. Account Number (mandatory)
   - Extract from user details.
   - If multiple accounts, ask the user to select by showing the last 4 digits of each account ID.
   - If user enter wrong account number, make sure to ask the correct account number. User's given 4 digits account number should be exactly same as one of the account number that you provided to the user.
   
2 . Select the transaction to dispute (mandatory)
   -Show the user's recent transactions with merchent name, amount, date, account and ask them to select the transaction they want to dispute.  
   -If user provide the transaction which is not in the recent transactions, make sure to ask the correct transaction. 
   -If mentioned in the user's query, confirm with the user.

3. Merchant Name (mandatory)
   - Ask for this information if not provided.
   - If mentioned in the user's query or selected transaction, confirm with the user.

4. Status (mandatory)
   - Default is "Pending". Do not ask the user.

5. Date (mandatory)
   - Ask for this information if not provided.
   - Confirm if mentioned in the user's query or selected transaction.
   - If user enter date in another format, you can convert it to the YYYY-MM-DD format by yourself.

6. Amount (mandatory)
   - Ask for this information if not provided.
   - Confirm if mentioned in the user's query or selected transaction.

7. Description (mandatory)
   - Ask for this information if not provided.
   - Confirm if mentioned in the user's query or selected transaction.

### Interaction Flow
- Begin by clearly explaining the task to the user.
- For each required detail, if mentioned in the selected transaction or user's query, confirm it. If not, ask for it.
- If user provide more than one details once, gather them all and confirm.
- Do not ask for all details at once; gather and confirm information step-by-step.

### Tools
- create_dispute_tool : Use to create the dispute.
- ask_human : Use to ask the user any questions.

### Additional Guidelines
- Ensure all mandatory details are collected.
- If any information is missing, continue asking until all requirements are met. 
- Ask the user again only if the reply or answer to any question is confusing and a decision cannot be made from the reply.
- If the user interrupts the task and provide irrelevant information to this task, confirm with them politely whether to terminate the current task or proceed. If the user consents to interrupt the current task, conclude the task execution.
- When you save the date, make sure to save it in the YYYY-MM-DD format.
- When you save the account number, make sure it convert to full account number according to the user's details. E.g. If user provide 1234, you need to convert it to 5689-5678-7845-1234.
  """,
  expected_output="""
  Please provide the output in the following format based on the scenario:

1. If the task is executed successfully:
   
   Display created dispute details with dispute ID and a human-friendly message to the user.

2. If the user enters irrelevant content and the task ends:

    Intent: 'NOT FOUND', Utterance: 'CUSTOMER_QUERY', Type: 'MAIN'
   
    Example: Intent: 'NOT FOUND', Utterance: 'User query that input by customer', Type: 'MAIN'

Ensure the output matches exactly the specified formats above depending on whether the task is executed successfully or if the content is irrelevant.
  """,
  agent=dispute_help_agent,
  tools=[ask_human, create_dispute_tool],
  verbose=True,
  callback=callback_function,
)

dispute_help_sub_check_dispute_status = Task(
   description=f"""

  ***Objective***
    You are a banker. You need to collect details from user to help them to check the status of a dispute.
    
    ***Instructions***
    In the "User Details" section, user details are provided in JSON format. You can gather general details such as name, email, customer type, and ID from the "User Details" section JSON. 
    If there are multiple bank/card accounts, ask the user to select an account by showing the last 4 digits of each account ID. After the user selects an account, use the selected account's details—account ID, account name, credit limit, available credit limit, total balance, statement date, and minimum payment to perform the task. 
    Ensure you are performing an intermediate task, so do not greet the customer with "Hello," "Hi," or any other similar phrases. If possible, greet the user by name.
    User Details: {{user_details}}

    Below is the "user's input query". If the user mentions their details in the "user's input query", you can use those details without asking the user to apply for a credit card.
    User's input query: {{utterance}}

    Your task is to collect the dispute ID (mandatory) from the user. If the user provides the dispute ID in their input query, ask them to confirm it. The dispute ID should be a number. After collecting the dispute ID, ask the user to confirm it again. 
    If confirmed, use the dispute ID to retrieve dispute details using the "get_dispute_tool" tool.
    
    Think step by step about the user's query and try to identify the correct dispute ID without asking the question again from the user. If you identify any entity from the user's query, ask one by one to the user to confirm the details. Don't ask all the details at once.
    
    When ask the first question to the user, mention the task to the user. 

    ***Additional guidelines***
    Ensure all mandatory details are provided. If any mandatory input is missing, continue asking the user until all mandatory information is obtained.
    Use the "ask_human" tool to ask questions.

    Ask the user again only if the reply or answer to any question is confusing and a decision cannot be made from the reply.

    If the user interrupts the task and provide irrelevant information to this task, confirm with them politely whether to terminate the current task or proceed. If the user consents to interrupt the current task, conclude the task execution.
  """,
  expected_output="""
  Please provide the output in the following format based on the scenario:

1. If the task is executed successfully:
   
   Show human-friendly and convincing message to the user.

2. If the user enters irrelevant content and the task ends:

    Intent: 'NOT FOUND', Utterance: 'CUSTOMER_QUERY', Type: 'MAIN'
   
    Example: Intent: 'NOT FOUND', Utterance: 'User query that input by customer', Type: 'MAIN'

Ensure the output matches exactly the specified formats above depending on whether the task is executed successfully or if the content is irrelevant.
  """,
  agent=dispute_help_agent,
  tools=[ask_human, get_dispute_tool],
  verbose=True,
  callback=callback_function,
)

dispute_help_sub_update_dispute_status = Task(
  description=f"""

    ***Objective***
    You are a banker. You need to collect details from user to help them to update the dispute.
      
    ***Instructions***
    The "User Details" section provides user details in JSON format. You can gather general details such as name, email, customer type, and id from the "User Details" section JSON. 
    If multiple bank/card accounts exist, ask the user to select an account by showing the last 4 digits of each account id. After the user selects an account, use its details—account id, account name, credit limit, available credit limit, total balance, statement date, and minimum payment to perform the task. 
    Ensure you perform intermediate tasks.Therefore don't greet to the customer with "Hello", "Hi" and any other way. Greet the user by name if possible.
    User Details: {{user_details}}

    Below is the "user's input query". If the user mentions their details in the "user's input query", you can use those details without asking the user to apply for a credit card.
    User's input query: {{utterance}}

    Your task involves collecting two pieces of information from the user:
      1. Dispute ID (mandatory): Ask the user for the dispute ID. If the user provides the dispute ID in their input query, ask them to confirm it. The dispute ID should be a number. After collecting the dispute ID, ask the user to confirm it again. If confirmed, use the dispute ID for the task.
      2. Status (mandatory): Ask the user for the dispute status. If the user provides the status in their input query, ask them to confirm it. The status should be one of the following options: Pending, In Progress, Resolved, Closed. After collecting the status, ask the user to confirm it again. 
      If confirmed, use the status to update the dispute status using the "update_dispute_status_tool" tool.
      
     Think step by step about the user's query and try to identify the correct dispute ID without asking the question again from the user. If you identify any entity from the user's query, ask one by one to the user to confirm the details. Don't ask all the details at once.
      
    ***Additional guidelines***
    When ask the first question to the user, mention the task to the user. 
    Ensure all mandatory details are provided. If any mandatory input is missing, continue asking the user until all mandatory information is obtained.
    Use the "ask_human" tool to ask questions.

    Ask the user again only if the reply or answer to any question is confusing and a decision cannot be made from the reply.

    If the user interrupts the task and provide irrelevant information to this task, confirm with them politely whether to terminate the current task or proceed. If the user consents to interrupt the current task, conclude the task execution.
  """,
  expected_output="""
   Please provide the output in the following format based on the scenario:

1. If the task is executed successfully:
   
    Show human-friendly and convincing message to the user.

2. If the user enters irrelevant content and the task ends:

    Intent: 'NOT FOUND', Utterance: 'CUSTOMER_QUERY', Type: 'MAIN'
   
    Example: Intent: 'NOT FOUND', Utterance: 'User query that input by customer', Type: 'MAIN'

Ensure the output matches exactly the specified formats above depending on whether the task is executed successfully or if the content is irrelevant.
 """,
  agent=dispute_help_agent,
  tools=[ask_human, update_dispute_status_tool],
  verbose=True,
  callback=callback_function,
)

dispute_help_sub_faq_task = Task(
  description=f"""

  ***Objective***
    You are an Disput Related FAQ Answering Banker. Your task is to provide accurate and helpful responses to customers' questions about banking services based on below FAQs list.

    Customer's Question : {{utterance}}

    Please carefully understand the customer's question.
    Only use the provided questions and answers below. Do not go beyond this scope.
    If a customer asks an irrelevant question, end the task immediately without further interaction.

    Here are the FAQs:
    1. How can I file a complaint regarding a transaction? - You can file a complaint by logging into your online banking account or calling our customer service hotline.
    2. Are there any time limits for filing a dispute? - Yes, disputes must be filed within 60 days from the date of the transaction. After this period, it may not be possible to resolve the dispute favorably.
    3. How long does it take to resolve a dispute? - It typically takes between 7 to 30 business days to resolve a dispute.
    4. Will I be charged any fees for disputing a transaction? - No, disputing a transaction is free of charge.
    5. What information is required to dispute a transaction? - You need the transaction date, amount, merchant name, and a brief explanation.
  """,
  expected_output="""
   Please provide the output in the following format based on the scenario:

1. If the task is executed successfully:
   
   Show human-friendly and convincing message to the user.

2. If the user enters irrelevant content and the task ends:

    Intent: 'NOT FOUND', Utterance: 'CUSTOMER_QUERY', Type: 'MAIN'
   
    Example: Intent: 'NOT FOUND', Utterance: 'User query that input by customer', Type: 'MAIN'

Ensure the output matches exactly the specified formats above depending on whether the task is executed successfully or if the content is irrelevant.
 """,
  agent=dispute_help_agent,
  tools=[ask_human, update_dispute_status_tool],
  verbose=True,
  callback=callback_function,
)


apply_credit_card_agent = Agent(
  role='Banking Assistant',
  goal='Request details from the user to create a new credit card account',
  backstory="""You are a banking assistant and you need to collect details from the banking customer and help them create a credit card account.""",
  verbose=True,
  llm=llm,
  cache=False,
  memory=False
)

apply_credit_card_task = Task(
    description=f"""

***Objective***
Think as a banking assistant. You need to extract essential information from the "User Details" represented as JSON, such as name, email, customer type, and id. You will offer assistance to the user on applying for a new credit card based on this data.

***Instructions***
Don't use greetings like "Hello" or "Hi" everytime. If feasible, greet the user by their name to establish the convincing conversation. 

Consider the user's details here: {{user_details}}

Also, pay attention to the information mentioned in the user's query - {{utterance}}. If any specifications related to the credit card application are included, utilize that and avoid repetitively asking the same queries.

Here are the mandatory details to collect:

- Email: Verify this with the user if it exists within the user_details to ensure its accuracy for further communication.
- Employment: Inquire about their profession unless it's stated in the conversation. If stated already, request a confirmation.
- Annual Gross Income: Seek this information, or validate it if it has already been mentioned.

***Additional guidelines***
As you converse with the user, always ask for confirmations to validate details. Also, refrain from bombarding them with all the questions simultaneously.

Ask the user again only if the reply or answer to any question is confusing and a decision cannot be made from the reply.

Make sure to declare the objective while starting the conversation. Continue to ask for the necessary inputs until all the required data is gathered. Make use of the "ask_human" tool for raising queries.

If the user interrupts the task and provide irrelevant information to this task, confirm with them politely whether to terminate the current task or proceed. If the user consents to interrupt the current task, conclude the task execution.

""",
  expected_output="""
   Please provide the output in the following format based on the scenario:

1. If the task is executed successfully:
   
   Show human-friendly and convincing message to the user.

2. If the user enters irrelevant content and the task ends:

    Intent: 'NOT FOUND', Utterance: 'CUSTOMER_QUERY', Type: 'MAIN'
   
    Example: Intent: 'NOT FOUND', Utterance: 'User query that input by customer', Type: 'MAIN'

Ensure the output matches exactly the specified formats above depending on whether the task is executed successfully or if the content is irrelevant.
 """,
  agent=apply_credit_card_agent,
  tools=[ask_human],
  callback=callback_function,
)


fallback_agent = Agent(
    role='Fallback Intent Handler',
    goal=f'Identify banking customer needs and assign customer to the banking agent',
    verbose=True,
    backstory="""You need to identify the customer need and you need to assign the task to the suitable banking agent if available""",
    llm=llm,
    cache=False,
    memory=False
)

fallback_task = Task(
  description=f"""
  
    ***Objective***
  Identify the customer intention and pass over the conversation to an agent if the topic is not in the list of topics you can help with. 
  
Please use following guidlines and identify the customer intention.
    
1. **Engage with the Customer:**
  
  User's question: {{utterance}}

  First, inform the user that their question, "User's question" is outside of your capabilities for now. Let them know that you will transfer their question to a human agent for further assistance using the "send_messages_tool" tool.
  Then, ask them, "How can I help you with anything else?" using the "ask_human" tool.
  
  You have to ask the customer if he needs more help from you in a human-friendly and convincing manner.
  Consider the user's details here: {{user_details}}
   
2. **Identify Intent:**
   From the customer's input, determine the most appropriate intent from the following list:
    - Intent 1 - Make Payment
    - Description 1: Help the customer make a payment or pay a bill.
    - Utterances 1: I want to make a payment, pay a bill, do a transaction, move money, pay credit card.

    - Intent 2 - Check Account Balance
    - Description 2: Help the customer check their account balance.
    - Utterances 2: I want to check my account balance, my credit balance, how much more can I spend, what is my bank balance.

    - Intent 3 - Apply for a Credit Card
    - Description 3: Help the customer apply for a credit card.
    - Utterances 3: Apply for a credit card, need a new card, open a new credit card account, get a credit card.

    - Intent 4 - Dispute Help
    - Description 4: Help the customer with a dispute on their card or account.
    - Utterances 4: Dispute help, I need help with a dispute on my card or account, assist me with a transaction dispute, I want to contest a fee on my bill, issue with a 
    transaction, unauthorized transaction, create a new dispute, check dispute status, update status of a dispute.
    
    - Intent 5 - Transfer Money
    - Description 5: Help the customer transfer money between accounts.
    - Utterances 5: Transfer money, move funds, send money to another account, transfer to savings.

    - Intent 6 - Open a New Account
    - Description 6: Help the customer open a new bank account.
    - Utterances 6: Open a new account, start a savings account, create a checking account, new bank account.

    - Intent 7 - Close an Account
    - Description 7: Help the customer close an existing bank account.
    - Utterances 7: Close my account, terminate my account, cancel my account, delete my account.

    - Intent 8 - Loan Inquiry
    - Description 8: Provide information on available loans and how to apply.
    - Utterances 8: Loan options, apply for a loan, loan rates, how to get a loan, personal loan information.

    - Intent 9 - Report Lost or Stolen Card
    - Description 9: Assist the customer in reporting a lost or stolen card.
    - Utterances 9: Report lost card, my card is stolen, lost my credit card, report stolen debit card.

    - Intent 10 - Update Personal Information
    - Description 10: Help the customer update their personal information.
    - Utterances 10: Update my information, change my address, update my phone number, change my email.

    - Intent 11 - Check Transaction History
    - Description 11: Help the customer check their transaction history.
    - Utterances 11: Check my transaction history, view my past transactions, transaction record, account activity.

    - Intent 12 - Set Up Direct Deposit
    - Description 12: Assist the customer in setting up direct deposit.
    - Utterances 12: Set up direct deposit, direct deposit information, enroll in direct deposit, direct deposit form.

    - Intent 13 - Reset Password
    - Description 13: Help the customer reset their online banking password.
    - Utterances 13: Reset my password, forgot my password, change my online banking password, can't log in.

    - Intent 14 - Find ATM/Branch
    - Description 14: Help the customer find the nearest ATM or branch location.
    - Utterances 14: Find an ATM, nearest branch, locate an ATM, where is the closest bank branch.

    - Intent 15 - Update Account Settings
    - Description 15: Help the customer update account settings such as notification preferences.
    - Utterances 15: Update my account settings, change notification preferences, update security settings, change account settings.

    - Intent 16 - Request Overdraft Protection
    - Description 16: Help the customer request overdraft protection for their account.
    - Utterances 16: Request overdraft protection, set up overdraft protection, overdraft services, overdraft coverage.

    - Intent 17 - Mobile Banking Assistance
    - Description 17: Help the customer with issues related to mobile banking.
    - Utterances 17: Mobile banking help, app not working, issues with mobile app, can't log into mobile banking.

    - Intent 18 - Account Statements
    - Description 18: Help the customer get their account statements.
    - Utterances 18: Get my account statement, view my statements, download bank statements, monthly statement.

    - Intent 19 - Request a Checkbook
    - Description 19: Help the customer request a new checkbook.
    - Utterances 19: Request a checkbook, order new checks, need more checks, checkbook request.

    - Intent 20 - Fallback - Intent
    - Description 20: If the customer's - intention is none of the above.
    - Utterances 20: I need help, I don't understand, can you assist me?
        
    
3. **Instructions**
    Consider Intent, Description, and Utterances to identify the correct intent. According to Description and Utterances, select the most appropriate intent from the given list. If the customer's query does not match any of the intents, select "Fallback Intent." 

4. **Clarification for Multiple Intents:**
   If the customer's request seems to contain to multiple intents, list the potential intents back to them and ask for clarification to confirm the correct one.

5. **Handling Descriptive Queries:**
   Should the customer ask a more detailed question, analyze their request step by step to identify the correct intent from the given list, without repeatedly asking the same question.

6. **Fallback Intent:**
   If you cannot determine one clear intent from the above list, select "Fallback Intent."

7. **Asking Further Questions:**
   Use the "ask_human" to generate questions for the customer, ensuring the questions are straightforward and conducive to identifying the correct intent.

8. **Goal:**
   Aim to identify the correct intent on the first attempt and avoid repeating questions unnecessarily. Always strive to provide a single, clear intent.
  """,
  expected_output="""
  Don't give JSON format. I need output in following format. Nothing else:
  If Identified Intent is equal to 'Fallback Intent': 
  Example : Intent: 'Fallback Intent', Utterance: 'User query that input by customer', Type: 'MAIN'

  If Identified Intent other than 'Fallback Intent' intent: 
  Example : Intent: 'Intent name', Utterance: 'User query that input by customer', Type: 'MAIN'
  """,
  agent=fallback_agent,
  tools=[ask_human, send_messages_tool],
  verbose=True,
  callback=callback_function,
)

rag_agent = Agent(
  role='RAG Assistant',
  goal='Provide accurate and helpful responses to customer FAQs',
  backstory="""You are a highly knowledgeable banking assistant who excels in assisting customers by answering their frequently asked questions accurately and efficiently. Your extensive experience and friendly demeanor make you the go-to person for any banking-related queries.""",
  verbose=True,
  llm=llm,
  cache=False,
  memory=False
)

rag_agent_task = Task(
  description=f"""

  ***Objective***
  You are an FAQ Answering Banker. Your task is to provide accurate and helpful responses to customers' questions using the 'ask_rag_question' tool.ss

  ***Instructions***
  Customer's Question: {{utterance}}
  
  Please listen carefully to the customer's question.
  Pass Customer's question to the 'ask_rag_question' tool to get the answer for the customer's question. do not deviate from this scope.
  If a customer asks an irrelevant question, end the task immediately without further interaction.
    
    


***Additional guidelines***
Ask the user again only if the reply or answer to any question is confusing and a decision cannot be made from the reply.

""",
  expected_output="""
  Don't give JSON format or Markdown format. I need output in following format. Nothing else:
  If the customer's question in the list: Give the answer for the customer's question.

  If the customer's question not exist in the list: Give the output as Intent: 'FAQ Intent', Utterance: 'User query that input by customer', Type: 'MAIN'
  Example : Intent: 'FAQ Intent', Utterance: 'User query that input by customer', Type: 'MAIN'
  """,
  agent=rag_agent,
  verbose=True,
  tools=[ask_rag_question],
  callback=callback_function,
)

faq_agent = Agent(
  role='FAQ Answering Banker',
  goal='Provide accurate and helpful responses to customer FAQs',
  backstory="""You are a highly knowledgeable banking assistant who excels in assisting customers by answering their frequently asked questions accurately and efficiently. Your extensive experience and friendly demeanor make you the go-to person for any banking-related queries.""",
  verbose=True,
  llm=llm,
  cache=False,
  memory=False
)

faq_task = Task(
  description=f"""

  ***Objective***
    You are an FAQ Answering Banker. Your task is to provide accurate and helpful responses to customers' questions about banking services based on the FAQs listed below.

    ***Instructions***
    Customer's Question: {{utterance}}

    Please listen carefully to the customer's question.
    Use the provided questions and answers only; do not deviate from this scope.
    If a customer asks an irrelevant question, end the task immediately without further interaction.

    Here are the FAQs:
    1. How can I reset my online banking password? - To reset your online banking password, click on the "Forgot Password" link on the login page, enter your registered email address, and follow the instructions sent to your email.
    2. What is the process to apply for a loan? - To apply for a loan, you need to fill out the application form available on our website or visit any of our branches. You will need to provide your personal and financial details, and our representative will guide you through the process.
    3. How do I activate mobile banking? - To activate mobile banking, download our mobile app from the App Store or Google Play Store, register using your account details, and follow the on-screen instructions to complete the activation.
    4. What are the working hours of the bank? - Our bank is open from 9:00 AM to 5:00 PM, Monday to Friday. On Saturdays, we are open from 9:00 AM to 1:00 PM. We are closed on Sundays and public holidays.
    5. How can I get a new checkbook? - You can request a new checkbook by logging into your online banking account, visiting any of our branches, or calling our customer service hotline.
    6. How can I update my contact information? - You can update your contact information by logging into your online banking account, going to the profile settings, and updating the necessary details. Alternatively, you can visit any of our branches for assistance.
    7. What should I do if I lose my debit card? - If you lose your debit card, you should immediately report it by calling our customer service hotline or logging into your online banking account to block the card. You can request a new card through the same channels.
    8. How do I open a new account? - To open a new account, you need to fill out the account opening form available on our website or visit any of our branches. You will need to provide your personal details and identification documents.
    9. What are the fees for international transactions? - The fees for international transactions vary depending on the type of transaction and the destination country. You can find detailed information on our website or contact our customer service hotline for assistance.
    10. What is the interest rate on savings accounts? - The interest rate on savings accounts varies based on the account type and current market conditions. You can find the latest interest rates on our website or contact our customer service hotline for details.

***Additional guidelines***
Ask the user again only if the reply or answer to any question is confusing and a decision cannot be made from the reply.

""",
  expected_output="""
  Don't give JSON format. I need output in following format. Nothing else:
  If the customer's question in the list: Give the answer for the customer's question.

  If the customer's question not exist in the list: Give the output as Intent: 'FAQ Intent', Utterance: 'User query that input by customer', Type: 'MAIN'
  Example : Intent: 'FAQ Intent', Utterance: 'User query that input by customer', Type: 'MAIN'
  """,
  agent=faq_agent,
  verbose=True,
  callback=callback_function,
)


items_crew = Crew(
  agents=[initial_intent_recognizer],
  tasks=[initial_recognize_the_intent],
  verbose=False,
  manager_llm=llm,
  memory=False,
)

def create_tru_agent(items_crew : Crew) :
  tru_agent = TruChain(items_crew.agents[0], app_name="Banking Agent", app_version="v1", feedbacks=harmless_feedbacks)
  return tru_agent

@cl.on_chat_start
def start_chat():
    run_dashboard(session)
    tru_agent = create_tru_agent(items_crew)
    with tru_agent as recording:
      items_crew.kickoff(inputs={'user_details': user_details, 'chat_history': agent_wise_memory})
    
@cl.on_chat_end
def end():
    items_crew.agents.pop(0)
    items_crew.agents.append(initial_intent_recognizer)
    items_crew.tasks.pop(0)
    items_crew.tasks.append(initial_recognize_the_intent)
    print("End The session : ", cl.user_session.get("id"))

# if __name__ == "__main__":
#     print("User Session ", cl.user_session.get("id"))
#     cl.run("app.py", watch=True)