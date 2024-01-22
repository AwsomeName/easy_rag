import streamlit as st
import pandas as pd
import numpy as np
from RAG import KnowLedge

session_stats = st.session_state

st.set_page_config(
   page_title="文档搜索问答",
   page_icon="📝",
   layout="wide",
   initial_sidebar_state="expanded",
)


try:
    kl = st.session_state.kl
except:
    pass

# 文件夹目录
global_dir = "data_pdf/data1"
colh1, colh2 = st.columns(2)
with colh1:
    # selected_directory = st.text_input('Enter the directory path:', '/path/to/your/directory')
    global_dir = st.text_input('Enter the directory path:', global_dir)
with colh2:
    st.write("数据读取目录是：" + global_dir)
    if st.button("更新目录"):
        kl.reset_folder(global_dir)
        print("update done")


df = pd.DataFrame(
   np.random.randn(50, 20),
   columns=('col %d' % i for i in range(20)))


col1, col2 = st.columns(2)

with col1:
    st.header("👇在这里输入问题")
    input_str = st.text_input(label="文本输入", placeholder="输入想要提问的内容, 回车键键提交", max_chars=1000)
    if st.button("提问！"):
        if input_str is not None and len(input_str) >0:
            # output_str, output_df = kl.search_result(input_str)
            output_df = kl.search_result(input_str)
            st.session_state['output_df_folder'] = output_df
            with st.expander(label="生成结果", expanded=True):
                with st.empty():
                    # st.markdown(output_str)
                    st.session_state['output_df_folder'] = output_df
                    for response, histroy in kl.stream_search():
                        # st.write(response)
                        st.markdown(response)
    # st.text_area(label="展示生成内容", placeholder="", height=600)


with col2:
    st.header("参考依据")
    if st.session_state.get('output_df') is not None:

        st.dataframe(st.session_state.get('output_df_folder'))  # Same as st.write(df)
    else:
        st.markdown("""
        ## 说明：
        1. 在左侧输入`问题`，然后按提问.
        2. 右上角会有`running`字样，表示程序正在运行.
        3. 结束后，会出现文本提取结果和对应的参考依据.
            - 3.1. 左下角文本框是生成的文本.
            - 3.2. 右侧是文本生成所参考的文档.
        """)