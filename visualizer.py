import streamlit as st
# NLP Pkgs
import spacy_streamlit
import spacy

nlp = spacy.load('output/model-last')


def main():
    """A Simple NLP app with Spacy-Streamlit"""
    st.title("Custom clinical Name Entity Recognizer (NER)")
    menu = ["NER"]
    choice = st.sidebar.selectbox("Menu", menu)
    # st.subheader("Named Entity Recognition")
    raw_text = st.text_area("Your Text", "Enter Text Here")
    docx = nlp(raw_text)
    spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)


if __name__ == '__main__':
    main()
