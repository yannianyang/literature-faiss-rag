import os
import re
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from openai import OpenAI
import numpy as np

# 清理非法代理字符
def clean_surrogate_content(raw_text: str) -> str:
    if not raw_text:
        return ""
    clean_str = ''.join([c for c in raw_text if not (0xD800 <= ord(c) <= 0xDFFF)])
    clean_str = re.sub(r"\n{2,}", "\n", clean_str)
    clean_str = clean_str.strip()
    return clean_str

# 加载环境变量
load_dotenv()
ENV_API_KEY = os.getenv("OPENAI_API_KEY")
ENV_API_BASE = os.getenv("OPENAI_API_BASE")
DB_PATH = "faiss_db"
PDF_FOLDER = "papers"
vector_db = None
openai_client = None

def build_knowledge_base():
    global vector_db
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
    all_docs = []
    pdf_files = []
    log_info = ""

    for file in os.listdir(PDF_FOLDER):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER, file)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            has_bad_char = False

            # 逐个清洗页面文本
            for doc in docs:
                origin = doc.page_content
                doc.page_content = clean_surrogate_content(doc.page_content)
                if len(origin) != len(doc.page_content):
                    has_bad_char = True

            if has_bad_char:
                tip = f"⚠️ {file} 存在异常字符，已自动清洗\n"
                print(tip.strip())
                log_info += tip

            all_docs.extend(docs)
            pdf_files.append(file)

    if not all_docs:
        return "【系统提示】未检测到有效PDF文档，请将文件放入papers目录后重试"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)
    # 分片二次兜底清洗
    for chunk_doc in split_docs:
        chunk_doc.page_content = clean_surrogate_content(chunk_doc.page_content)

    client = OpenAI(api_key=ENV_API_KEY, base_url=ENV_API_BASE)
    texts = [doc.page_content for doc in split_docs]
    embeddings_list = []
    for text in texts:
        resp = client.embeddings.create(input=text, model="text-embedding-v2")
        embeddings_list.append(np.array(resp.data[0].embedding))

    vector_db = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings_list)),
        embedding=None
    )
    vector_db.save_local(DB_PATH)

    res_text = f"{log_info}【构建成功】共加载文档{len(pdf_files)}份，生成文本分片{len(split_docs)}块，向量库已持久化保存"
    return res_text

def chat_answer(question, input_key, input_base, model_name):
    global vector_db, openai_client
    api_key = ENV_API_KEY if ENV_API_KEY else input_key
    api_base = ENV_API_BASE if ENV_API_BASE else input_base
    if not api_key:
        return "请配置API密钥（.env 文件 或 页面输入框）"

    if openai_client is None:
        openai_client = OpenAI(api_key=api_key, base_url=api_base)

    if vector_db is None:
        if os.path.exists(DB_PATH):
            vector_db = FAISS.load_local(
                DB_PATH,
                embeddings=None,
                allow_dangerous_deserialization=True
            )
        else:
            return "请先上传PDF并构建知识库"

    clean_question = clean_surrogate_content(question)
    resp = openai_client.embeddings.create(input=clean_question, model="text-embedding-v2")
    query_embedding = np.array(resp.data[0].embedding)
    docs_and_scores = vector_db.similarity_search_by_vector(query_embedding, k=6)
    context = "\n\n".join([doc.page_content for doc in docs_and_scores])

    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=api_base,
        model_name=model_name,
        temperature=0.6
    )
    template = """你是一个专业的文档问答助手，只能基于下面提供的文档内容回答问题。
    如果文档里没有相关信息，请直接说“文档中未找到相关信息”，不要编造答案。
    文档内容：{context}
    问题：{question}
    """
    prompt = template.format(context=context, question=clean_question)
    response = llm.invoke(prompt)
    return response.content

# UI
with gr.Blocks(title="大模型知识库问答系统") as demo:
    gr.Markdown("# 大模型知识库问答系统")
    with gr.Row():
        model_select = gr.Dropdown(
            label="选择模型",
            choices=["qwen-turbo", "qwen-plus"],
            value="qwen-turbo"
        )
        api_key_input = gr.Textbox(label="API 密钥", placeholder="可填写，优先读取.env配置")
        api_base_input = gr.Textbox(label="API 接口地址", placeholder="可填写，优先读取.env配置")
    build_btn = gr.Button("上传并构建知识库", variant="primary")
    build_result = gr.Textbox(label="构建日志")
    build_btn.click(build_knowledge_base, outputs=build_result)
    question_input = gr.Textbox(label="提问", placeholder="请输入基于文档的问题")
    answer_output = gr.Textbox(label="回答", lines=8)
    submit_btn = gr.Button("开始提问")
    submit_btn.click(
        fn=chat_answer,
        inputs=[question_input, api_key_input, api_base_input, model_select],
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)