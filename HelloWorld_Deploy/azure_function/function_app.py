import azure.functions as func
import logging
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_elasticsearch import ElasticsearchStore
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

app = func.FunctionApp()

load_dotenv(verbose=True)
logging.info("Starting application initialization...")

CONFIG_NAME = "mongo_config.json"
logging.info(f"## config_name : {CONFIG_NAME}")

with open(f'configs/{CONFIG_NAME}', 'r') as f:
    config = json.load(f)

if config['db'] == 'elasticsearch':
    os.environ["ES_CLOUD_ID"] = os.getenv("ES_CLOUD_ID")
    os.environ["ES_USER"] = os.getenv("ES_USER")
    os.environ['ES_PASSWORD'] = os.getenv("ES_PASSWORD")
    os.environ["ES_API_KEY"] = os.getenv("ES_API_KEY")
elif config['db'] == 'mongo':
    os.environ["MONGODB_ATLAS_CLUSTER_URI"] = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
    logging.info(f'## db : {config["db"]}')
    logging.info(f'## db_name : {config["path"]["db_name"]}')

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

# Load DB once at startup
db = None
try:
    logging.info("Starting database initialization...")
    if config['db'] == 'elasticsearch':
        db = ElasticsearchStore(
            index_name='helloworld',
            embedding=OpenAIEmbeddings()
        )
    elif config['db'] == 'mongo':
        client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"], ssl=True)
        MONGODB_COLLECTION = client[config['path']['db_name']][config['path']['collection_name']]
        db = MongoDBAtlasVectorSearch(
            collection=MONGODB_COLLECTION,
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            index_name=config['path']['index_name'],
            relevance_score_fn="cosine"
        )
    else:
        raise ValueError("Wrong db value set in config file")
    logging.info("Database initialized successfully.")
except Exception as e:
    logging.error(f"Error loading database: {str(e)}")

# 여기에 generate_ai_response 함수 정의를 추가하세요
def generate_ai_response(conversation_history, query, db):
    llm = ChatOpenAI(
        model=config['openai_chat_inference']['model'],
        frequency_penalty=config['openai_chat_inference']['frequency_penalty'],
        logprobs=config['openai_chat_inference']['logprobs'],
        top_logprobs=config['openai_chat_inference']['top_logprobs'],
        max_tokens=config['chat_inference']['max_new_tokens'],
        temperature=config['chat_inference']['temperature'],
    )

    template_text = """
    당신은 한국의 외국인 근로자를 위한 법률 및 비자 전문 AI 어시스턴트입니다. 다음 지침을 따라 응답해 주세요:

    1. 관련 문서의 정보를 바탕으로 정확하고 최신의 법률 및 비자 정보를 제공하세요.
    2. 복잡한 법률 용어나 절차를 쉽게 설명하여 외국인 근로자가 이해하기 쉽게 답변하세요.
    3. 불확실한 정보에 대해서는 명확히 언급하고, 공식 기관에 문의할 것을 권장하세요.
    4. 문화적 차이를 고려하여 정중하고 친절한 태도로 응대하세요.
    5. 필요한 경우 관련 정부 기관이나 지원 센터의 연락처를 제공하세요.
    6. 개인정보 보호를 위해 구체적인 개인 정보를 요구하지 마세요.
    7. 이전 대화 내용을 참고하여 문맥에 맞는 자연스러운 응답을 제공하세요.
    8. 사용자의 이전 질문이나 concerns를 기억하고 연관된 정보를 제공하세요.

    관련 문서: 
    {context}

    대화 기록:
    {conversation_history}
    """

    try:
        similar_docs = db.similarity_search(query, k=3)
        for i, doc in enumerate(similar_docs):
            logging.info(f"Top-{i+1} document : {doc.page_content}")

        context = " ".join([doc.page_content for doc in similar_docs])
        prompt_template = PromptTemplate.from_template(template_text)
        filled_prompt = prompt_template.format(context=context, conversation_history=conversation_history)
        
        output = llm.invoke(input=filled_prompt)
        return output.content
    except Exception as e:
        logging.error(f"Error in generate_ai_response: {str(e)}")
        raise

@app.route(route="question", auth_level=func.AuthLevel.ANONYMOUS)
def question(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Question function triggered.")
    
    if db is None:
        logging.error("Database is not initialized.")
        return func.HttpResponse("Database not initialized", status_code=500)
    
    try:
        req_body = req.get_json()
        logging.info(f"Received data: {req_body}")

        conversation = req_body.get('Conversation', [])
        if not conversation:
            return func.HttpResponse("No conversation data provided", status_code=400)

        user_query = next((item['utterance'] for item in reversed(conversation) if item['speaker'] == 'human'), None)
        if user_query is None:
            return func.HttpResponse("No user utterance found", status_code=400)

        logging.info(f"Extracted user query: {user_query}")

        answer = generate_ai_response(conversation, user_query, db)
        logging.info(f"Generated AI response: {answer}")

        return func.HttpResponse(answer, status_code=200)

    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        return func.HttpResponse(f"An error occurred: {str(e)}", status_code=500)

@app.route(route="get_test/{param}", auth_level=func.AuthLevel.ANONYMOUS)
def get_echo_call(req: func.HttpRequest) -> func.HttpResponse:
    param = req.route_params.get('param')
    return func.HttpResponse(json.dumps({"param": param}), mimetype="application/json")