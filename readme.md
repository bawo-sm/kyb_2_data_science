# Ogólnie
Repo z notatkami, eksperymentami, skryoptami data science dotycżaczymi projektu KYB


# Uruchomienie Ubuntu 24
1. Wirtualne środowisko python
```
python3.12 -m venv venv
source venv/bin/activate
```
2. Instalacja pakietów
```
pip install uv
uv pip install -r requirements.txt
```


# Mozliwe strategie ML (od najłatwiejszych)
1. Sprowadzenie informacji o spręecie do danych tabelarycznych (metadane, wymiary) + trening modelu drzewaistego (np. lgbm, xgboost, random forest)
2. Sieci neuronowe i klasyfikacja
    - warstwy konwolucyjne 3d (mentalnie proste ale wymaga wnajwięcej GPU)
    - architektura PointNet (złożona architektura)
3. Sieci neuronowe i embeddingi: 
    - jesli danych będzie mało, to można transformowac sprzęt do psotaci embeddingów
    - wtedy utworzy się "bazę wiedzy" z embeddingami oraz ich relacjami, np. która część, z która współpracuje dobrze
    - model można wytrenować, np. w podejściu *contrastive learning*


# Przydatne linki
0. Repo z projekty KYB 1 https://github.com/cloudstateu/KYB
1. Implementacja sieci PointNet
    - pytorch: https://medium.com/@sepideh.92sh/decoding-pointnet-a-practical-guide-to-3d-segmentation-with-python-and-pytorch-7a037fecb8a7
    - keras & tensorflow: https://keras.io/examples/vision/pointnet/
2. Implementacja conv3d z pytorch: https://medium.com/data-science/pytorch-step-by-step-implementation-3d-convolution-neural-network-8bf38c70e8b3
3. PointNet z hugging face https://huggingface.co/keras-io/pointnet_segmentation
