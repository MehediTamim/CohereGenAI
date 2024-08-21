from rest_framework import serializers

class ClassificationInputSerializer(serializers.Serializer):
    input_text = serializers.CharField()
    model = serializers.CharField()
    csv_file_path = serializers.CharField()
    text_column = serializers.CharField()
    label_column = serializers.CharField()

class EmbeddingSearchSerializer(serializers.Serializer):
    text = serializers.CharField()
    model_name = serializers.CharField()
    csv_path = serializers.CharField()
    collection_name = serializers.CharField()