import streamlit as st
from RAG import KnowLedge


session_stats = st.session_state


@st.cache_resource
def create_model():
    kl = KnowLedge(gen_model_name_or_path="models/chatglm3-6b-32k",
                   sen_embedding_model_name_or_path="models/chinese-roberta-wwm-ext")
    return kl

# kl = create_model()
# kl = st.session_state.kl

st.set_page_config(
    page_title="你好",
    page_icon="👋",
)

st.write("# 欢迎! 👋")

# st.sidebar.success("在上方选择一个演示。")

st.markdown(
    """
    这是一个基于大模型的文档问答框架。
    **👈 从侧边栏选择一个功能，开始大模型问答之旅吧！
    file QA, 支持选择、上传文件，进行问答。
    folder_QA, 读取一个文件夹内的所有文件，进行搜索问答。
"""
)

st.session_state.kl = create_model()