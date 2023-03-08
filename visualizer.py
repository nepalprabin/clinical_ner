import streamlit as st
# NLP Pkgs
import spacy_streamlit
import spacy

nlp = spacy.load('en_pipeline')


def main():
    """A Simple NLP app with Spacy-Streamlit"""
    st.title("Custom clinical Name Entity Recognizer (NER)")
    menu = ["NER"]
    choice = st.sidebar.selectbox("Menu", menu)
    # st.subheader("Named Entity Recognition")
    raw_text = st.text_area("Your Text", "A 52-year-old male is presented to the emergency department with complaints of chest pain and shortness of breath. His medical history is significant for hypertension, hyperlipidemia, and a prior myocardial infarction. On physical examination, he appeared anxious and diaphoretic with a blood pressure of 150/90 mmHg, heart rate of 110 beats per minute, and respiratory rate of 24 breaths per minute. An electrocardiogram showed ST-segment elevation in leads II, III, and aVF, and he was diagnosed with acute coronary syndrome. He was started on aspirin, heparin, and clopidogrel, and transferred to the cardiac catheterization lab for further management.")
    docx = nlp(raw_text)
    spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)


if __name__ == '__main__':
    main()
