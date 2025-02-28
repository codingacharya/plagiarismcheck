import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

def check_plagiarism(uploaded_text, reference_texts):
    texts = [uploaded_text] + reference_texts
    vectorizer = TfidfVectorizer().fit_transform(texts)
    similarity_matrix = cosine_similarity(vectorizer)
    similarity_scores = similarity_matrix[0][1:]
    return similarity_scores

def main():
    st.title("Plagiarism Checker")
    st.write("Upload a text file to check for plagiarism against reference documents.")
    
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    
    reference_texts = [
        "This is a sample document to check plagiarism against.",
        "Another example reference document for testing text similarity.",
        "Plagiarism detection is essential in academic and professional writing."
    ]
    
    if uploaded_file is not None:
        uploaded_text = uploaded_file.read().decode("utf-8")
        st.text_area("Uploaded Text", uploaded_text, height=200)
        
        similarity_scores = check_plagiarism(uploaded_text, reference_texts)
        
        results_df = pd.DataFrame({
            "Reference Document": [f"Document {i+1}" for i in range(len(reference_texts))],
            "Similarity Score": similarity_scores
        })
        
        st.subheader("Plagiarism Check Results")
        st.write(results_df)
        
        max_score = max(similarity_scores)
        if max_score > 0.7:
            st.error("High similarity detected! Potential plagiarism.")
        elif max_score > 0.4:
            st.warning("Moderate similarity detected. Review suggested.")
        else:
            st.success("Low similarity. No major issues detected.")
    
if __name__ == "__main__":
    main()
