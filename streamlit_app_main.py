import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Função para carregar os dados do CSV. Usamos o decorador cache_data para cache de dados.
@st.cache_data
def load_data():
    df = pd.read_csv('merged_data_v2.csv', usecols=['class_concat', 'abstract', 'totalwords'])
    df_filtered = df.loc[df['class_concat'] == 'D']
    sentences = df_filtered['abstract'].tolist()
    classes = df_filtered['class_concat'].tolist()
    return sentences, classes

# Função para carregar o modelo de embeddings. O decorador cache_resource é usado para recursos globais como modelos.
@st.cache_resource
def load_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    return model

# Função para carregar os embeddings dos dados. O decorador cache_data é adequado pois embeddings são dados específicos.
@st.cache_data
def load_embeddings():
    parts = []
    for i in range(1, 7):
        parts.append(np.load(f'df_embeddings_d_part_{i}.npy'))
    embeddings = np.concatenate(parts)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings_norm

# Função para calcular e retornar as frases mais similares com base na entrada do usuário
def get_similar_sentences(input_sentence, model, embeddings_d_norm, df_sentences_d, classes_d, top_k=5):
    sentence_embedding = model.encode(input_sentence)
    sentence_embedding_norm = sentence_embedding / np.linalg.norm(sentence_embedding)
    cosine_similarities = np.dot(embeddings_d_norm, sentence_embedding_norm)
    top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]
    top_similarities = cosine_similarities[top_indices]
    results = [(df_sentences_d[idx], classes_d[idx], top_similarities[i] * 100) for i, idx in enumerate(top_indices)]
    return results

# Função principal que cria a interface do usuário no Streamlit
def main():
    st.title("Busca de Frases Semelhantes")

    # Definição de uma chave para o campo de entrada para manter o estado
    user_input_key = "user_input"
    user_input = st.text_area("Digite sua ideia:", value=st.session_state.get(user_input_key, ""), height=150, key=user_input_key)

    # Botões para interação
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Buscar Similaridades"):
            if user_input:
                # Carregamento dos dados e modelos necessários
                df_sentences_d, classes_d = load_data()
                model = load_model()
                embeddings_d_norm = load_embeddings()

                results = get_similar_sentences(user_input, model, embeddings_d_norm, df_sentences_d, classes_d)
                for sentence, classe, score in results:
                    with st.expander(f"Score: {score:.2f}% - Classe: {classe}"):
                        st.write(sentence)
    with col2:
        if st.button("Limpar"):
            st.session_state[user_input_key] = ""
            st.rerun()

if __name__ == "__main__":
    main()
