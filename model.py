# import os
# import numpy as np
# import faiss
# import nltk
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# import nltk
# nltk.download('punkt')



# embeddings = np.load("embeddings (1).npy")
# filenames = np.load("filenames.npy")
# # Load FAISS index and filenames
# faiss_index = faiss.read_index("faiss_index (1).index")

# # Function to find the most similar file
# def find_most_similar_file(new_file_path, faiss_index, model):
#     with open(new_file_path, 'r', encoding='utf-8') as file:
#         new_text = file.read().strip()
#     # Create embedding for the new file
#     new_embedding = model.encode(new_text, convert_to_tensor=True).reshape(1, -1)

#     # Search for the most similar file
#     distances, indices = faiss_index.search(new_embedding, k=1)
#     most_similar_index = indices[0][0]
#     similarity_score = distances[0][0]
#     most_similar_filename = filenames[most_similar_index]
    
#     return most_similar_filename, similarity_score, most_similar_index
# # Provide the new file path
# new_file_path = "cleaned_26-sep-24-ceh-day2_transcription.txt"
# most_similar_file, similarity_score, most_similar_index = find_most_similar_file(new_file_path, faiss_index, model)

# # Print the most similar file along with the similarity score
# print(f"Most similar file: {most_similar_file}")
# print(f"Similarity score: {similarity_score:.4f}")
# print(f"Index: {most_similar_index}")


# # Function to split text into sentences
# def split_into_sentences(text):
#     return nltk.sent_tokenize(text)

# # Function to classify sentences as relevant, irrelevant, or flagged
# def classify_sentences(new_file_sentences, similar_file_sentences, model, top_n=3):
#     relevant_sentences = []
#     irrelevant_sentences = []
#     flagged_sentences = []

#     new_embeddings = [model.encode(sentence, convert_to_tensor=False) for sentence in new_file_sentences]
#     similar_embeddings = [model.encode(sentence, convert_to_tensor=False) for sentence in similar_file_sentences]

#     for i, new_embedding in enumerate(new_embeddings):
#         similarities = cosine_similarity([new_embedding], similar_embeddings)[0]
#         # Calculate average similarity from top-N most similar sentences
#         top_similarities = np.sort(similarities)[-top_n:]
#         avg_similarity = np.mean(top_similarities)

#         # Dynamic threshold tuning
#         if avg_similarity >= 0.7:  # Adjust as needed based on dataset
#             relevant_sentences.append((new_file_sentences[i], avg_similarity))
#         elif avg_similarity <= 0.5:
#             irrelevant_sentences.append((new_file_sentences[i], avg_similarity))
#         else:
#             flagged_sentences.append((new_file_sentences[i], avg_similarity))
    
    
#     # Return grouped sentences
#     return relevant_sentences, irrelevant_sentences, flagged_sentences

# # Function to compare the most similar file with the new file
# def compare_files(new_file_path, most_similar_file, model, folder_path="C:\\Users\\DELL\\Desktop\\CYS_project\\audio"):
#     with open(new_file_path, 'r', encoding='utf-8') as file:
#         new_text = file.read().strip()
#     with open(os.path.join(folder_path, most_similar_file), 'r', encoding='utf-8') as file:
#         similar_text = file.read().strip()

#     new_file_sentences = split_into_sentences(new_text)
#     similar_file_sentences = split_into_sentences(similar_text)

#    # Classify and group sentences
#     relevant_sentences, irrelevant_sentences, flagged_sentences = classify_sentences(new_file_sentences, similar_file_sentences, model)

#     return relevant_sentences, irrelevant_sentences, flagged_sentences

# # Compare new file with the most similar file and classify sentences
# relevant_sentences, irrelevant_sentences, flagged_sentences = compare_files(new_file_path, most_similar_file, model)

# # Save grouped classifications to a text file
# # Save grouped classifications to a text file
# output_file = "classification_results2.txt"
# with open(output_file, "w", encoding="utf-8") as file:
#     file.write("Relevant Sentences:\n")
#     file.write("-------------------\n")
#     for sentence in relevant_sentences:
#         file.write(f"- {sentence[0]}\n")  # Add bullet point
#     file.write("\nIrrelevant Sentences:\n")
#     file.write("---------------------\n")
#     for sentence in irrelevant_sentences:
#         file.write(f"- {sentence[0]}\n")  # Add bullet point

#     file.write("\nFlagged Sentences:\n")
#     file.write("------------------\n")
#     for sentence in flagged_sentences:
#         file.write(f"- {sentence[0]}\n")  # Add bullet point

# print(f"Results saved to {output_file}")

import os
import numpy as np
import faiss
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load your pre-trained model and necessary data
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = np.load("embeddings (1).npy")
filenames = np.load("filenames.npy")
faiss_index = faiss.read_index("faiss_index (1).index")

nltk.download('punkt', quiet=True)

# Path to the folder containing the comparison files
COMPARISON_FOLDER = 'C:\\Users\\VARSHINI\\OneDrive\\Desktop\\cybersercurity\\COMPARISON_FOLDER\\audio'

def split_into_sentences(text):
    return nltk.sent_tokenize(text)

def classify_sentences(new_file_sentences, similar_file_sentences, top_n=3):
    relevant_sentences = []
    irrelevant_sentences = []
    flagged_sentences = []

    new_embeddings = [model.encode(sentence, convert_to_tensor=False) for sentence in new_file_sentences]
    similar_embeddings = [model.encode(sentence, convert_to_tensor=False) for sentence in similar_file_sentences]

    for i, new_embedding in enumerate(new_embeddings):
        similarities = cosine_similarity([new_embedding], similar_embeddings)[0]
        top_similarities = np.sort(similarities)[-top_n:]
        avg_similarity = np.mean(top_similarities)

        if avg_similarity >= 0.7:  # Relevant threshold
            relevant_sentences.append((new_file_sentences[i], avg_similarity))
        elif avg_similarity <= 0.5:  # Irrelevant threshold
            irrelevant_sentences.append((new_file_sentences[i], avg_similarity))
        else:
            flagged_sentences.append((new_file_sentences[i], avg_similarity))

    return relevant_sentences, irrelevant_sentences, flagged_sentences

def find_most_similar_file(new_file_path):
    logger.debug(f"Reading file: {new_file_path}")
    with open(new_file_path, 'r', encoding='utf-8') as file:
        new_text = file.read().strip()
    new_embedding = model.encode(new_text, convert_to_tensor=True).reshape(1, -1)

    distances, indices = faiss_index.search(new_embedding, k=1)
    most_similar_index = indices[0][0]
    most_similar_filename = filenames[most_similar_index]
    
    return most_similar_filename, most_similar_index

def compare_files(new_file_path, most_similar_file):
    logger.debug(f"Comparing new file: {new_file_path}")
    logger.debug(f"With similar file: {os.path.join(COMPARISON_FOLDER, most_similar_file)}")
    
    with open(new_file_path, 'r', encoding='utf-8') as file:
        new_text = file.read().strip()
    with open(os.path.join(COMPARISON_FOLDER, most_similar_file), 'r', encoding='utf-8') as file:
        similar_text = file.read().strip()

    new_file_sentences = split_into_sentences(new_text)
    similar_file_sentences = split_into_sentences(similar_text)

    return classify_sentences(new_file_sentences, similar_file_sentences)

def process_file(file_path):
    logger.debug(f"Processing file: {file_path}")
    most_similar_file, _ = find_most_similar_file(file_path)
    relevant, irrelevant, flagged = compare_files(file_path, most_similar_file)

    return relevant, irrelevant, flagged