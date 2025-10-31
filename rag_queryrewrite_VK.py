import os
import openai
import azure.identity
import csv
from dotenv import load_dotenv
from lunr import lunr



# get access to .env environment variables
load_dotenv( override=True)
api_host = os.getenv("API_HOST", "gitHub")

# define a function that will get the LLM Client for one of these model providers Azure, OpenAI, ollama, GitHub

def getOpenAIClient(pApiHost:str) -> tuple[openai.OpenAI, str]:
    client : openai.OpenAI
    MODEL_NAME : str = ""
    if pApiHost == "azure" :
        client = openai.OpenAI(base_url=os.getenv("AZURE_OPENAI_ENDPOINT"), api_key=azure.identity.DefaultAzureCredential())
        MODEL_NAME = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    elif pApiHost == "ollama":
        client = openai.OpenAI(base_url=os.environ["OLLAMA_ENDPOINT"], api_key="")
        MODEL_NAME = os.environ["OLLAMA_MODEL"]
    elif pApiHost == "openai" :
        client = openai.OpenAI(api_key=os.environ["OPENAI_KEY"])
        MODEL_NAME = os.environ["OPENAI_MODEL"]
    elif pApiHost == "github":
        client = openai.OpenAI( base_url=os.environ["GITHUB_ENDPOINT"], api_key=os.environ["GITHUB_TOKEN"])
        MODEL_NAME = os.environ["GITHUB_MODEL"]
    return client, MODEL_NAME

llm_client, MODEL_NAME =getOpenAIClient(api_host)



# retrieve the RAG csv into a list
with open(mode= "r", file="hybrid.csv") as csvFile :
    reader = csv.reader(csvFile)
    rows = list(reader)

# print(rows)
document = [{"id":(i), "body":" ".join(row)} for i, row in enumerate(rows[1:], start=1)]
# print(document)
# create a lunr index so we can do a search of the RAG list based upon the question and only get recods that are relevant to the question
indexed_RAG_data = lunr(ref="id", fields=["body"], documents=document)
# print(indexed_RAG_data)

# def search_RAG_return_ids( pQuery:str, pIndexed_RAG_data:Index) -> str :
#     #search the indexed RAG data
#     pIndexed_RAG_data.
def search_RAG(pQuery:str) -> str :
    index_result = indexed_RAG_data.search(pQuery)
    result_rows = [rows[int(row["ref"])] for row in index_result]
    retstr = "|".join(rows[0]) #first header row
    retstr += "\n"+"|".join("---" for _ in range(len(rows[0]))) # ---|---|---... generate the table header underline
    # now append the result rows
    retstr += "\n"+"\n".join("|".join(row) for row in result_rows)
    return retstr



assistant_query_rewriter_system_prompt = f"""You are an assistant who rewrites user question into a keywords search 
query that can be used to search an index having the columns {",".join(rows[0])}.
A good keyword search qury will only contain keywords as words in a sentance without any punctuation. 
Give a string containing only the keywords without any formatting."""
assistant_system_prompt = f"""You are an assistant who is knowledgeful in hybrid vehicles. 
You will only use the provided Source as the source of information to respond to the user question"""
prompt_messages = [{"role":"system", "content":assistant_system_prompt}]
# print(assistant_query_rewriter_system_prompt)
# becasue this is a multiturn rewrite, we need a endless loop
while True :
    # prompt the user to get their question
    user_question = input("Enter your query about Hybrid Vehicles: ")
    #send that question to the llm for it to give us the query ( role: system, content: you are helpful assistant who will read the user question and provides only the key words for searching. )
    response = llm_client.chat.completions.create(model=MODEL_NAME, temperature=0.5, messages= [
        {"role": "system", "content": assistant_query_rewriter_system_prompt},
        {"role":"user", "content": f"New user question: {user_question} \n Conversation history: {prompt_messages}"}
    ])
    llm_keyword_query_response = response.choices[0].message.content
    print(llm_keyword_query_response)
    # use that query to get the data from the loaded RAG list
    # index_result = indexed_RAG_data.search(llm_keyword_query_response)
    # print(index_result)
    RAG_result = search_RAG(llm_keyword_query_response)
    prompt_messages.append( {"role":"user", "content":f"""User Question: {user_question}. Source:{RAG_result} """})
    response = llm_client.chat.completions.create( model=MODEL_NAME, temperature=0.5, messages=prompt_messages)
    prompt_messages.append({"role":"assistant", "content":response.choices[0].message.content})
    print(f"""Response from {api_host} Model {MODEL_NAME} is
          >>>>>>>> {response.choices[0].message.content} <<<<<<<<<<<<""")

#   send this  