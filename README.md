# Pracownia Chemii Kwantowej - Model Sztucznej Inteligencji

Projekt realizowany w ramach pracowni rotacyjnej "Chemia Kwantowa" przez Jana Wolskiego, Marzec 2025.
Pełny raport z przygotowani projektu w `raport.pdf` <br>
Enaglish version of README.md <u>[here](ang_README.md)</u>

## Cel projektu
Celem projektu jest stworzenie modelu sztucznej inteligencji, który na podstawie deskryptorów substancji chemicznych przewiduje powinowactwo elektronowe podanego związku chemicznego.

## Źródło danych
Dane pozyskane za pomocą scrapera internetowego napisanego w Go, korzystającego z biblioteki gorod. Źródłem danych jest [NIST Chemistry WebBook](https://webbook.nist.gov). Dane zapisywane są w formacie CSV zawierającym nazwę związku, wartość powinowactwa elektronowego (EA) oraz wzór sumaryczny.

## Deskryptory chemiczne
Projekt wykorzystuje 32 deskryptory chemiczne, takie jak:
- Liczba atomów w cząsteczce
- Liczba atomów poszczególnych pierwiastków (np. H, C, N)
- Masa cząsteczkowa
- Promienie jonowe, elektroujemności Paulinga, polaryzowalności, energie jonizacji
- Hydrogen Deficiency Index (HDI)

Pełną listę deskryptorów znajdziesz w pliku `descryptors.py`.

## Model
Rozbudowany model wykorzystujący warstwy konwolucyjne (CNN), mechanizm uwagi (Self-Attention) oraz klasyczne warstwy Dense:
- MAE = 0.1415
- RMSE = 0.2205
- R² = 0.935

## Uruchomienie modelu
1. Pobierz lub sklonuj repozytorium.
2. Pobierz lub wygeneruj dane przy pomocy scrapera.
3. Uruchom skrypt `descryptors.py`, by wygenerować deskryptory.

### Wytrenowanie modelu
Aby użyć gotowego wytrenowanego modelu do przewidywania powinowactwa elektronowego, użyj:
```bash
python main.py --model <typ_modelu> --action train
```

Przykład:
```
python main.py --model class --action train
```

### Użycie gotowego modelu
Aby użyć gotowego wytrenowanego modelu do przewidywania powinowactwa elektronowego, użyj:
```bash
python main.py --model <typ_modelu> --action use --target <wzór_chemiczny>
```

Przykład:
```
python main.py --model reg --action use --target "C7H13-"
```

### Rodzaje --model
* Regresyjny na całych wartościach - `reg`
* Regresyjnych na tylko dodatnich wartościach - `preg`
* Clasyfikator dla EA > 3.6 eV - `class`

### Rodzaje --action
* Użycie gotowego modelu z folderu `src` - `use`
* Trenowanie modelu - `train`

## Struktura projektu
- `descryptors.py` – obliczanie deskryptorów chemicznych
- `models.py` – implementacja sieci neuronowej
- `main.py` – skrypt do używania modelu
- `src` - tam znajdują się pliki modeli: tokenizera, scalera i modeli z projektu


## Wykorzystane biblioteki
- [PyTorch](https://pytorch.org)
- [scikit-learn](https://scikit-learn.org)
- [pandas](https://pandas.pydata.org)
- [mendeleev](https://github.com/lmmentel/mendeleev)

## Potencjalne kierunki rozwoju
- Zwiększenie liczby danych
- Integracja autoenkodera w celu poprawy jakości predykcji
- Rozbudowa zestawu deskryptorów

## Źródła
- [NIST Chemistry WebBook](https://webbook.nist.gov)
- [PyTorch](https://pytorch.org)
- [scikit-learn](https://scikit-learn.org)


