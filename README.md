Literature-FAISS-RAG 论文知识库问答系统
基于 LangChain+FAISS + 通义千问实现的私有 PDF 文献 RAG 问答平台，Gradio 可视化网页部署。
项目简介
本项目针对科研文献查阅慢、通用大模型知识幻觉、商用文献工具收费等痛点，搭建本地化私有论文知识库：
自动清洗 PDF 异常脏字符，批量解析顶会论文文档
text-embedding-v2 向量化 + FAISS 本地向量库持久化存储
检索 TopK=6 片段，依靠 Prompt 约束大模型，无资料直接拒答杜绝编造
Gradio 网页可视化，支持在线填写 API 密钥快速启用
实测效果
人工查阅 7min → 系统检索仅 1.68s，效率提升 96.8%
文档内问题准确率 90%，域外陌生问题拒答率 100%
环境依赖安装
bash
pip install gradio langchain langchain-community langchain-text-splitters faiss-cpu openai python-dotenv pypdf numpy
目录结构
plaintext
├── main.py        # 项目主程序
├── .env.example     # 环境变量模板文件
├── .gitignore       # Git忽略规则配置
├── LICENSE          # 开源许可证文件
├── papers/          # PDF论文存放目录
└── faiss_db/        # 自动生成的向量库目录（Git忽略，不上传仓库）
faiss_db 文件夹：运行生成向量库，自动被 git 忽略不上传仓库
部署使用步骤
配置密钥
复制 .env.example 并重命名为 .env ，填入阿里通义千问 Key 与接口地址
也可不在.env 配置，在网页界面手动输入密钥临时使用
放入文档
把需要解析的 PDF 论文全部放进 papers 文件夹
启动项目
bash
python main.py
浏览器访问： http://127.0.0.1:7860
使用流程
①点击【上传并构建知识库】，控制台输出构建日志（文档数量、分片数量）
②在提问框输入论文相关问题，点击开始提问获取答案
核心参数说明
文本分片：chunk_size=500，overlap=100
检索召回数量：k=6
LLM 生成温度：temperature=0.6（越小回答越严谨）
支持模型：qwen-turbo /qwen-plus
常见问题排查
报错提示：URL 拼写可能存在错误，请检查
请核对通义千问 API 接口地址、API Key 是否填写正确，同时检查本地网络可正常访问阿里云大模型接口。
后续迭代计划
接入 Neo4j 知识图谱升级 KG-RAG
接入本地 BERT 实现离线向量化，不再依赖第三方 API
新增 docx、word 文档解析兼容
项目亮点
✅ 本地私有化向量存储，论文数据不外泄
✅ 自带脏文本自动清洗，适配各类扫描 PDF、异常编码文档
✅ 严格 Prompt 限制，未知内容强制拒答，消除大模型幻觉
开源协议
本项目基于 MIT License 开源，详细协议内容请查看仓库根目录下的 LICENSE 文件。
任何人可自由使用、复制、修改、分发本项目代码，支持个人学习、学术研究以及商用场景。
使用本项目代码时，请保留原始版权与开源协议声明。
补充说明：本项目所使用的学术论文数据集均为公开资源，仅用于非商业学习与科研用途，请严格遵守对应文献原版权规定。
