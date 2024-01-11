import streamlit as st
import pandas as pd
import numpy as np
import docx
from tqdm import tqdm
from demo import KnowLedge



st.set_page_config(
   page_title="文档搜索问答",
   page_icon="📝",
   layout="wide",
   initial_sidebar_state="expanded",
)

@st.cache_resource
def create_model(global_dir):
    # kl = KnowLedge(global_dir=global_dir)
    kl = KnowLedge(global_dir="data/data1",
                   gen_model_name_or_path="models/chatglm3-6b-32k",
                   sen_embedding_model_name_or_path="models/chinese-roberta-wwm-ext")
    return kl

# 文件夹目录
global_dir = "data_pdf/data1"
kl = create_model(global_dir)

# streamlit run web_ui.py --server.fileWatcherType none
df = pd.DataFrame(
   np.random.randn(50, 20),
   columns=('col %d' % i for i in range(20)))

# @st.cache_resource
st.header("👇在这里选择上传的文件，目前只有docx格式")
new_upload_file = st.file_uploader("docx", type=['docx'])
# upload_file = "data_pdf/data1"
upload_file = None
if new_upload_file:
    upload_file = new_upload_file

if upload_file:
    print(upload_file)
    kl.reset_file(upload_file, 'docx')
    # doc = docx.Document(upload_file)
    # all_paras = doc.paragraphs
    # raw_text = [i.text for i in tqdm(all_paras)]
    # print("raw_text:", raw_text)
    print("load done")
    st.stop()


col1, col2 = st.columns(2)

with col1:
    st.header("👇在这里输入问题")
    input_str = st.text_input(label="文本输入", placeholder="输入想要提问的内容, 回车键键提交", max_chars=100)
    if input_str is not None and len(input_str) >0:

        output_str, output_df = kl.search_result(input_str)
        st.session_state['output_df'] = output_df
        with st.expander(label="生成结果", expanded=True):
            st.markdown(output_str)
    # st.text_area(label="展示生成内容", placeholder="", height=600)


with col2:
    st.header("参考依据")
    if st.session_state.get('output_df') is not None:

        st.dataframe(st.session_state.get('output_df'))  # Same as st.write(df)
    else:
        st.markdown("""
        ## 说明：
        1. 在左上角输入`问题`，然后按`enter`提问.
        2. 右上角会有`running`字样，表示程序正在运行.
        3. 结束后，会出现文本提取结果和对应的参考依据.
            - 3.1. 左下角文本框是生成的文本.
            - 3.2. 右侧是文本生成所参考的文档.
        """)