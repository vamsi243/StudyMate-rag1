from langchain.embeddings import HuggingFaceBgeEmbeddings

class BHE:  # Note: Class names are typically in PascalCase
    def __init__(self):
        self.model_name = "BAAI/bge-base-en"
        self.model_kwargs = {'device': 'cuda'}
        self.encode_kwargs = {'normalize_embeddings': True}
        self.model_norm = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )

# Create an instance of the class
bhe_model = BHE()
