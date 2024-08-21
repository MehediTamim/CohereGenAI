from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import cohere
from .serializers import ClassificationInputSerializer
from .serializers import EmbeddingSearchSerializer
from Apis.utils import load_csv_data_as_json, prepare_documents, load_embedding_model, get_milvus_instance, store_documents_in_vectordb, similarity_search

class ClassifyTextView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ClassificationInputSerializer(data=request.data)
        if serializer.is_valid():
            input_text = serializer.validated_data['input_text']
            model = serializer.validated_data['model']
            csv_file_path = serializer.validated_data['csv_file_path']
            text_column = serializer.validated_data['text_column']
            label_column = serializer.validated_data['label_column']
            # api_key = 'your_cohere_api_key'  # Replace with your actual API key
            api_key = "24NMCUKMYm1gUfDz21h5wESNicxxtsu36G1RZtqb"
            
            examples = self.load_csv_data_with_pandas(csv_file_path, text_column, label_column)
            cohere_client = self.initialize_cohere_client(api_key)
            classification_result = self.classify_text(cohere_client, model, input_text, examples)
            
            return Response({
                'prediction': classification_result[0].prediction,
                'confidence': classification_result[0].confidence
            }, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def load_csv_data_with_pandas(self, csv_file_path, text_column, label_column):
        import pandas as pd
        from cohere import ClassifyExample
        
        df = pd.read_csv(csv_file_path)[:50]
        examples = [
            ClassifyExample(text=row[text_column], label=row[label_column])
            for _, row in df.iterrows()
        ]
        return examples

    def initialize_cohere_client(self, api_key):
        import cohere
        return cohere.Client(api_key)

    def classify_text(self, cohere_client, model, input_text, examples):
        response = cohere_client.classify(
            model=model,
            inputs=[input_text],
            examples=examples
        )
        return response.classifications
    

class StoreDocumentsView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = EmbeddingSearchSerializer(data=request.data)
        if serializer.is_valid():
            csv_path = serializer.validated_data['csv_path']
            model_name = serializer.validated_data['model_name']
            api_key = "24NMCUKMYm1gUfDz21h5wESNicxxtsu36G1RZtqb"
            collection_name = serializer.validated_data['collection_name']

            json_data = load_csv_data_as_json(csv_path)
            documents, uuids = prepare_documents(json_data)
            embeddings = load_embedding_model(model_name, api_key)
            vector_store = get_milvus_instance(embeddings, "http://localhost:19530", collection_name)
            store_documents_in_vectordb(vector_store, documents, uuids)

            return Response({"message": "Documents stored successfully."}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SimilaritySearchView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = EmbeddingSearchSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            model_name = serializer.validated_data['model_name']
            api_key = "24NMCUKMYm1gUfDz21h5wESNicxxtsu36G1RZtqb"
            collection_name = serializer.validated_data['collection_name']

            embeddings = load_embedding_model(model_name, api_key)
            vector_store = get_milvus_instance(embeddings, "http://localhost:19530", collection_name)
            metadata, score = similarity_search(vector_store, text, k=1)

            return Response({"metadata": metadata, "score": score}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)