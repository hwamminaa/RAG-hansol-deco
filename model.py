from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from data_loader import load_train_data

data = load_train_data()

# Embedding Model 로드
modelPath = "distiluse-base-multilingual-cased-v1"
model_kwargs = {'device':'cuda:1'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


# Document Encoding
# db에 벡터 저장
## db = Chroma.from_documents(data, embedding=embeddings)
# db = Chroma.from_documents(data, embeddings, persist_directory="./chroma_db")
# db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

db = Chroma.from_documents(data, embeddings)


result = db.similarity_search("하자 발생 시 보수 작업은?")
print(result)

result = db.similarity_search_with_score("방청 페인트의 종류에는 어떤 것들이 있는지 알고 계신가요? 또한, 원목사이딩을 사용하는 것에 어떤 단점이 있을까요?")
print(result)



# Retriever
retriever = db.as_retriever(search_kwargs={"k": 4})


# Generator
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import LlamaForCausalLM, AutoTokenizer, pipeline

model_id = "beomi/llama-2-ko-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id).to(1)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512,device=1)
hf = HuggingFacePipeline(pipeline=pipe)

'''
MODEL = 'beomi/KoAlpaca-Polyglot-5.8B'
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
).to(1)
pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=1)
hf = HuggingFacePipeline(pipeline=pipe)
'''



from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """마지막에 질문에 답하려면 다음과 같은 맥락을 사용합니다.

{context}

질문: {question}

유용한 답변:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | hf
    | StrOutputParser()
)


# 예시 출력
for chunk in rag_chain.stream("도배지에 녹은 자국이 발생하는 주된 원인과 그 해결 방법은 무엇인가요?"):
    print(chunk, end="", flush=True)

for chunk in rag_chain.stream("방청 페인트의 종류에는 어떤 것들이 있는지 알고 계신가요? 또한, 원목사이딩을 사용하는 것에 어떤 단점이 있을까요?"):
    print(chunk, end="", flush=True)


# test
import pandas as pd
from tqdm import tqdm

result = []

test = pd.read_csv("/home/minahwang2001/data/test.csv")
for i in tqdm(range(len(test))):
  _id = test.at[i,'id']
  _q = test.at[i,'질문']
  _a = []
  for chunk in rag_chain.stream(_q):
      _a.append(chunk)
      print(chunk, end="", flush=True)
  result.append(
      {
          "id":_id,
          "대답":" ".join(_a)
      }
  )
  print()

pd.DataFrame(result).to_csv("/home/minahwang2001/baseline/results_llama2.csv",index=False)


# submission 결과 저장
import pickle
from sentence_transformers import SentenceTransformer

with open("result.pkl",'wb') as f:
  pickle.dump(result,f)

_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

for i in range(len(result)):
  result[i]['embedding'] = _model.encode(result[i]['대답'].replace("\u200b"," "))

submission = []

for i in range(len(result)):
  tmp = {"id":result[i]['id'],}
  for j in range(len(result[i]['embedding'])):
    tmp[f'vec_{j}'] = result[i]['embedding'][j]
  submission.append(
      tmp
  )

pd.DataFrame(submission).to_csv("/home/minahwang2001/baseline/submission_llama2.csv",index=False)