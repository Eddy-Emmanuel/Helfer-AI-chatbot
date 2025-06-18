import os
import tempfile
from model.schema import AnalystSchema
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from fastapi import UploadFile, HTTPException, status
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader

from config.secret_keys import project_config
os.environ["OPENAI_API_KEY"] = project_config.openai_api_key

llm = ChatOpenAI(model="gpt-o4-mini", temperature=0)
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

async def LoadFile(file: UploadFile):
    file_name = file.filename.lower()
    file_bytes = await file.read()

    try:
        if file_name.endswith((".png", ".jpg", ".jpeg")):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                loader = UnstructuredImageLoader(tmp_path)
                return loader.load()
            finally:
                os.remove(tmp_path)

        elif file_name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                loader = UnstructuredPDFLoader(tmp_path)
                return loader.load()
            finally:
                os.remove(tmp_path)

        else:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                detail="Unsupported file format")

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to load file: {str(e)}")
        
class AnalystUtil:
    def __init__(self, llm, schema, embedding, template):
        self.llm = llm
        self.schema = schema
        self.embedding = embedding
        self.template = template
    
    def InjestAgent(self, data, prompt):
        vectorestore_retriever = FAISS.from_documents(documents=data,
                                                      embedding=self.embedding)
        
        prompt = PromptTemplate(input_variables=["context", "question"],
                                template=self.template)
        
        retriever_qa = RetrievalQA.from_chain_type(llm.with_structured_output(AnalystSchema),
                                                   chain_type="stuff",
                                                   retriever=vectorestore_retriever.as_retriever(),
                                                   return_source_documents=False,
                                                   chain_type_kwargs={"prompt":prompt})
        