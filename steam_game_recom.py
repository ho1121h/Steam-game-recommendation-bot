import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('3. review 토큰화&임베딩 결과.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('오박사의 게임추천')
st.markdown("## 오늘의 게임은 뭘까요?")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('당신: ', '')
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.sort_values(by='distance', ascending=False)
    ans1, ans2, ans3, ans4, ans5, ans6 = answer['game_name'].drop_duplicates()[:6]
    ans = [ans1, ans2, ans3, ans4, ans5, ans6]
    best_ans = answer[:100].value_counts('game_name').idxmax()
    if best_ans in ans:
        ans.remove(best_ans)
    output = (f"추천 게임은 {best_ans}입니다. 다른 추천은 {ans[0]}, {ans[1]}, {ans[2]}, {ans[3]}, {ans[4]}이 있습니다")
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_박사bot')


#streamlit run steam_game_recom.py