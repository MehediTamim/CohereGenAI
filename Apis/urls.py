from django.urls import path
from .views import ClassifyTextView, StoreDocumentsView, SimilaritySearchView

urlpatterns = [
    path('classify/', ClassifyTextView.as_view(), name='classify-text'),
    path('store-documents/', StoreDocumentsView.as_view(), name='store-documents'),
    path('similarity-search/', SimilaritySearchView.as_view(), name='similarity-search'),
]