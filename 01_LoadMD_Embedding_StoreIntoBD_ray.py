from langchain_community.document_loaders import (PyPDFLoader)
#from langchain.text_splitter import (RecursiveCharacterTextSplitter)
import os
import numpy as np
from ltp import StnSplit
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma, Milvus
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)
import re
import ray
from hdfs import InsecureClient
import pickle
import tempfile

path_db = "data/ChromaDB"

#Choose the embedding model
#model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = "/home/codelformat/shared_models/bge-m3"

embedding_function = SentenceTransformerEmbeddings(model_name=model_name, model_kwargs={"device": "cuda"})

#here is the setting for the size of chunk, 100 is one article only one chunk
THRESHOLD = 70

class MarkdownParser:
    def __init__(self):
        # 标准 Markdown 表格模式
        self.table_pattern = re.compile(
            r'''
            (?:\n|^)                     
            (?:\|.*?\|.*?\|.*?\n)        
            (?:\|(?:\s*[:-]+[-| :]*\s*)\|.*?\n) 
            (?:\|.*?\|.*?\|.*?\n)+
            ''', re.VERBOSE)
        
        # 无边框表格模式
        self.no_border_table_pattern = re.compile(
            r'''
            (?:\n|^)                 
            (?:\S.*?\|.*?\n)
            (?:(?:\s*[:-]+[-| :]*\s*).*?\n)
            (?:\S.*?\|.*?\n)+
            ''', re.VERBOSE)

    def extract_content(self, markdown_text):
        # 提取表格
        tables = self.table_pattern.findall(markdown_text)
        remainder = self.table_pattern.sub('', markdown_text)
        
        # 提取无边框表格
        no_border_tables = self.no_border_table_pattern.findall(remainder)
        remainder = self.no_border_table_pattern.sub('', remainder)
        
        # 合并所有内容
        all_content = [remainder] + tables + no_border_tables
        return [content.strip() for content in all_content if content.strip()]

class SemanticParagraphSplitter:
    def __init__(self, threshold=THRESHOLD, model_path=model_name):
        self.threshold = threshold
        self.model = SentenceTransformer(model_path)
        self.markdown_parser = MarkdownParser()

    @staticmethod
    def cut_sentences(text):
        sentences = StnSplit().split(text)
        return sentences

    @staticmethod
    def combine_sentences(sentences, buffer_size=2):
        # Go through each sentence dict
        for i in range(len(sentences)):

            # Create a string that will hold the sentences which are joined
            combined_sentence = ''

            # Add sentences before the current one, based on the buffer size.
            for j in range(i - buffer_size, i):
                # Check if the index j is not negative (to avoid index out of range like on the first one)
                if j >= 0:
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += sentences[j]['sentence'] + ' '

            # Add the current sentence
            combined_sentence += sentences[i]['sentence']

            # Add sentences after the current one, based on the buffer size
            for j in range(i + 1, i + 1 + buffer_size):
                # Check if the index j is within the range of the sentences list
                if j < len(sentences):
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += ' ' + sentences[j]['sentence']

            # Then add the whole thing to your dict
            # Store the combined sentence in the current sentence dict
            sentences[i]['combined_sentence'] = combined_sentence

        return sentences

    def build_sentences_dict(self, sentences):
        indexed_sentences = [{'sentence': x, 'index': i} for i, x in enumerate(sentences)]
        combined_sentences = self.combine_sentences(indexed_sentences)

        embeddings = self.model.encode([x['combined_sentence'] for x in combined_sentences], normalize_embeddings=True)

        for i, sentence in enumerate(combined_sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]

        return combined_sentences

    @staticmethod
    def calculate_cosine_distances(sentences):
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['combined_sentence_embedding']
            embedding_next = sentences[i + 1]['combined_sentence_embedding']

            # Calculate cosine similarity
            # similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            similarity = embedding_current @ embedding_next.T
            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the list
            distances.append(distance)

            # Store distance in the dictionary
            sentences[i]['distance_to_next'] = distance

        # Optionally handle the last sentence
        # sentences[-1]['distance_to_next'] = None  # or a default value

        return distances, sentences

    def calculate_indices_above_thresh(self, distances):
        breakpoint_distance_threshold = np.percentile(distances, self.threshold)
        # The indices of those breakpoints on your list
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
        return indices_above_thresh

    @staticmethod
    def cut_chunks(indices_above_thresh, sentences):
        # Initialize the start index
        start_index = 0

        # Create a list to hold the grouped sentences
        chunks = []

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks

    def split(self, text):
        # 首先解析 Markdown
        if isinstance(text, str):
            content_blocks = self.markdown_parser.extract_content(text)
            print(f"Found {len(content_blocks)} content blocks")
            input("Press Enter to continue...")
        else:
            content_blocks = [text]

        all_chunks = []
        for block in content_blocks:
            single_sentences = self.cut_sentences(block)
            if single_sentences:
                chunks = self.split_passages(single_sentences)
                all_chunks.extend(chunks)
        
        return all_chunks

    def split_passages(self, passages):
        combined_sentences = self.build_sentences_dict(passages)
        distances, sentences = self.calculate_cosine_distances(combined_sentences)

        indices_above_thresh = self.calculate_indices_above_thresh(distances)
        chunks = self.cut_chunks(indices_above_thresh, sentences)
        return chunks

# Add HDFS client configuration
hdfs_client = InsecureClient('http://localhost:9870')  # Adjust HDFS URL as needed

# Initialize Ray
ray.init()


@ray.remote
def process_document(file_path, embedding_function):
    print(f"Processing file: {file_path}")
    
    # Read from HDFS
    if file_path.endswith('.pdf'):
        with hdfs_client.read(file_path) as reader:
            content = reader.read()
    else:  # For .md files
        with hdfs_client.read(file_path) as reader:
            content = reader.read().decode('utf-8')

    text_splitter = SemanticParagraphSplitter(threshold=THRESHOLD)
    docs = text_splitter.split(content)
    
    # Generate embeddings
    embeddings = embedding_function.encode(docs)
    
    # Save embeddings and docs to HDFS
    output_filename = os.path.basename(file_path) + '.pkl'
    output_path = f"Embeddings/{output_filename}"
    
    # Create temporary file
    with tempfile.NamedTemporaryFile() as tmp:
        pickle.dump({'docs': docs, 'embeddings': embeddings}, tmp)
        tmp.flush()
        # Upload to HDFS
        hdfs_client.upload(output_path, tmp.name)
    
    print(f"Successfully saved embeddings for {file_path} to HDFS")
    return output_path

def process_documents_in_folder(hdfs_input_path, embedding_function):
    # Ensure Embeddings directory exists in HDFS
    if not hdfs_client.status('Embeddings', strict=False):
        hdfs_client.makedirs('Embeddings')
    
    # List files from HDFS
    files = hdfs_client.list(hdfs_input_path)
    
    # Create Ray tasks for each file
    tasks = []
    for filename in files:
        if filename.endswith(('.pdf', '.md')):
            file_path = f"{hdfs_input_path}/{filename}"
            task = process_document.remote(file_path, embedding_function)
            tasks.append(task)
    
    # Wait for all tasks to complete
    embedding_paths = ray.get(tasks)
    return embedding_paths

def combine_embeddings_to_chroma(embedding_paths, path_db):
    all_docs = []
    all_embeddings = []
    
    # Read all embeddings from HDFS
    for path in embedding_paths:
        with tempfile.NamedTemporaryFile() as tmp:
            # Download from HDFS to temp file
            hdfs_client.download(path, tmp.name)
            # Load the data
            with open(tmp.name, 'rb') as f:
                data = pickle.load(f)
                all_docs.extend(data['docs'])
                all_embeddings.extend(data['embeddings'])
    
    # Create Chroma database
    db = Chroma.from_embeddings(
        embeddings=all_embeddings,
        texts=all_docs,
        persist_directory=path_db
    )
    print(f"Successfully created Chroma database with {len(all_docs)} documents")
    return db

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process documents from HDFS and store embeddings.")
    parser.add_argument("--hdfs_path", type=str, default="markdowns", 
                       help="The path in HDFS containing the documents")
    args = parser.parse_args()
    
    # Process files and save embeddings to HDFS
    embedding_paths = process_documents_in_folder(args.hdfs_path, embedding_function)
    
    # Combine all embeddings into Chroma
    db = combine_embeddings_to_chroma(embedding_paths, path_db)
    
    # Shutdown Ray
    ray.shutdown()