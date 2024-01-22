import streamlit as st
import pandas as pd
import numpy as np

session_stats = st.session_state

st.set_page_config(
   page_title="文档搜索问答",
   page_icon="📝",
   layout="wide",
   initial_sidebar_state="expanded",
)

df = pd.DataFrame(
   np.random.randn(50, 20),
   columns=('col %d' % i for i in range(20)))


if True:
    st.write("👇选择上传的文件")
    # st.header("👇选择上传的文件")
    new_upload_file = st.file_uploader("", type=['docx', 'pdf'])
    upload_file = None
    if new_upload_file:
        upload_file = new_upload_file
    kl = st.session_state.kl

    reset_file = False
    old_file = None
    if upload_file:
        print(upload_file)
        if upload_file != old_file:
            file_name = upload_file.name
            # print("file_name:", file_name, file_name[-5:])
            if file_name[-5:] == ".docx":
                # print("----------docx")
                kl.reset_file(upload_file, 'docx')
            else:
                # print("----------pdf")
                kl.reset_file(upload_file, 'pdf')
            reset_file = True
        else:
            reset_file = False
        print("upload file done")
        old_file = upload_file


hcol1, hcol2 = st.columns(2)
with hcol1:
    # st.header("👇在这里输入问题")
    input_str = st.text_input(label="👇在这里输入问题", placeholder="输入想要提问的内容", max_chars=400)
    
    # st.write("\n\n")
    if st.button("提问！"):
        if upload_file is not None:
            if input_str is not None and len(input_str) >0:
                # output_str, output_df = kl.search_result(input_str)
                output_df = kl.search_result(input_str)
                # output_df.delete("text")
                del output_df['text']
                st.session_state['output_df'] = output_df
                # with st.expander(label="生成结果", expanded=True):
                    # st.markdown(output_str)
                with st.expander(label="生成结果", expanded=True):
                    with st.empty():
                        for response, histroy in kl.stream_search():
                            st.markdown(response)
                
        else:
            st.markdown("请选择文件")

# col1, col2 = st.columns(2)

with hcol2:

    st.header("参考依据")
    if st.session_state.get('output_df') is not None:
        st.dataframe(st.session_state.get('output_df'))  
    else:
        st.markdown("""
        ## 说明：
        1. 在正上方选择文件
        2. 在左上角输入`问题`，然后点击提问摁钮.
        3. 右上角会有`running`字样，表示程序正在运行.
        4. 结束后，会出现文本提取结果和对应的参考依据.
            - 4.1. 左下角文本框是生成的文本.
            - 4.2. 右侧是文本生成所参考的文档.
        """)

# with col2:
    # pass