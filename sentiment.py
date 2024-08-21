from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
import logging
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("groq_api_key"), 
    model_name='mixtral-8x7b-32768'
)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

class EmailData(BaseModel):
    body: str

@app.post("/analyze-sentiment")
async def analyze_sentiment(data: EmailData):
    sentiment_template = PromptTemplate(
        input_variables=['chat_history', 'body'],
        template="""
        Using the context from memory {chat_history}, analyze the sentiment of the following email body:
        
        Body: {body}
        
        Provide a One word sentiment i.e. Positive, negative, neutral etc. of the email. Don't write anything else your response should be 1-3 words only.
        """
    )

    try:
        sentiment_chain = LLMChain(llm=llm, prompt=sentiment_template, verbose=True, output_key='output', memory=memory)
    except Exception as e:
        logging.error(f"Error creating LLMChain: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating LLMChain: {e}")

    try:
        response = sentiment_chain({
            'body': data.body
        })
    except Exception as e:
        logging.error(f"Error during LLMChain execution: {e}")
        raise HTTPException(status_code=500, detail=f"Error during LLMChain execution: {e}")

    return {"sentiment_analysis": response['output']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("your_script_name:app", host="0.0.0.0", port=8000, reload=True)
