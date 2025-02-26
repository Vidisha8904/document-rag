import re
from rank_bm25 import BM25Okapi
from keybert import KeyBERT

# Function to remove punctuation from text
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)  # Removes all punctuation except words & spaces

# Initialize KeyBERT
kw_model = KeyBERT()

def extract_keywords(query, top_n=3):
    """Extracts top_n keywords from the query using KeyBERT"""
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    extracted = [kw[0].lower() for kw in keywords] if keywords else query.lower().split()
    return extracted

# Read chunks from the text file
with open("chunks.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# Split chunks based on "Chunk X from QUIZ 2017-18.pdf:"
chunks = raw_text.split("Chunk")[1:]  # Ignore empty first split

# Process chunks into a list (removing punctuation)
documents = []
for chunk in chunks:
    lines = chunk.strip().split("\n")[1:]  # Ignore "X from QUIZ..." line
    clean_text = " ".join(lines)  # Combine lines into a single string
    clean_text = remove_punctuation(clean_text)  # Remove punctuation
    documents.append(clean_text)  # Store cleaned text

# Tokenize the documents (convert to lowercase and split into words)
tokenized_docs = [doc.lower().split() for doc in documents]

print("_______________________________________________________________")
print(tokenized_docs)  # Debugging output to check tokenized documents

# Initialize BM25
bm25 = BM25Okapi(tokenized_docs)

# Query
query = "what is Arthrology?"
extracted_keywords = extract_keywords(query)  # Extract important keywords
print(f"Extracted Keywords: {extracted_keywords}")  # Debugging output

# Ensure extracted keywords exist in documents before querying BM25
filtered_keywords = [word for kw in extracted_keywords for word in kw.split() if any(word in doc for doc in tokenized_docs)]

if not filtered_keywords:
    print("No relevant keywords found in documents!")
else:
    # Get BM25 similarity scores
    scores = bm25.get_scores(filtered_keywords)

    # Retrieve top 5 most similar documents
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

    # Print the top 5 ranked documents
    for idx in ranked_indices:
        print(f"Score: {scores[idx]:.4f}\nDocument:\n{documents[idx]}\n")
        print("\n" + "="*50 + "\n")  # Separator for readability
