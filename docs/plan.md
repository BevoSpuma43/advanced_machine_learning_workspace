# Piano di progetto: classificazione di feature reali e sintetiche

## 1. Scopo del documento

Questo documento e la specifica operativa per implementare il progetto universitario di Advanced Machine Learning. Deve essere usato come fonte di verita da persone o modelli LLM che scrivono il codice.

Il progetto confronta cinque classificatori binari e un ensemble:

1. SVM con kernel RBF;
2. XGBoost;
3. MLP;
4. CNN spatial head;
5. ConvNeXt feature classifier;
6. ensemble composto esclusivamente da SVM RBF, XGBoost e CNN spatial head.

La finalita principale e verificare se l'ensemble eseguibile in locale ottiene prestazioni comparabili con ConvNeXt, addestrato su Google Colab.

La possibilita di usare una DeconvNet e stata valutata, ma non fa parte dell'implementazione corrente perche le immagini originali target non sono disponibili. La relativa sezione resta come specifica di una possibile estensione futura.

## 2. Vincoli e hardware

Hardware locale:

- CPU Intel Core i7-9700;
- 16 GB RAM;
- NVIDIA GeForce GTX 1660 Super con 6 GB VRAM;
- sistema operativo Windows;
- ambiente Jupyter Notebook.

ConvNeXt deve poter essere addestrato su Google Colab. La CNN spatial head deve essere sufficientemente leggera da poter essere addestrata localmente.

Tutto il codice sperimentale deve essere contenuto in notebook Jupyter. Ogni notebook deve essere self-contained, eseguibile dall'inizio alla fine con kernel pulito e avere la stessa struttura logica. Non introdurre moduli Python esterni indispensabili all'esecuzione dei notebook.

Tutti i notebook, inclusi quelli destinati principalmente all'esecuzione locale, devono essere compatibili con Google Colab. Lo stesso file `.ipynb` deve poter funzionare sia in locale sia su Colab senza modifiche manuali alle celle di codice. Sono consentiti soltanto parametri di configurazione espliciti, per esempio il percorso del dataset o la directory di output.

## 3. Dataset disponibile

File:

- `dataset/features_0.npy`;
- `dataset/features_1.npy`.

Caratteristiche verificate:

| File | Shape | Tipo | Campioni | Dimensione |
|---|---:|---:|---:|---:|
| `features_0.npy` | `(1654, 1280, 16, 16)` | `float32` | 1654 | circa 2.02 GiB |
| `features_1.npy` | `(2383, 1280, 16, 16)` | `float32` | 2383 | circa 2.91 GiB |

Totale: 4037 campioni. Ogni campione contiene 327680 valori ed occupa circa 1.25 MiB.

Mapping delle classi confermato:

```text
features_0.npy -> label 0 -> real
features_1.npy -> label 1 -> fake
```

La classe positiva per tutte le metriche deve essere `fake`, cioe label 1. I 1654 campioni reali provengono da Pascal-VOC e COCO; i 2383 campioni fake provengono da DALL-E 3, Stable Diffusion 1.4, Stable Diffusion 1.5 e Midjourney.

Controlli gia eseguiti:

- nessun NaN;
- nessun valore infinito;
- nessun duplicato binario esatto interno alle classi;
- nessun duplicato binario esatto tra le due classi;
- classe maggioritaria: 2383/4037, baseline accuracy pari a circa 0.5903.

Le feature aggregate tramite global average pooling contengono gia un segnale discriminativo rilevante. Questo puo indicare sia reale separabilita sia bias di acquisizione/preprocessing. Il report deve quindi includere una discussione sui possibili confondenti.

### Informazioni ancora utili prima dell'addestramento definitivo

- conferma che indice `i` del tensore corrisponda stabilmente a uno specifico campione;
- eventuali ID di gruppo: immagine sorgente, prompt, generatore, seed, video o soggetto;

Il mapping delle classi deve essere dichiarato esplicitamente nella configurazione di tutti i notebook e non rideterminato automaticamente dal nome dei file.

## 4. Struttura prevista del repository

```text
project-root/
|-- dataset/
|   |-- features_0.npy
|   |-- features_1.npy
|   |-- tabular_features_gap.npy
|   |-- tabular_features_mean_std.npy
|   |-- spatial_channel_normalization.npz
|   `-- sample_index.csv
|-- docs/
|   `-- plan.md
|-- notebooks/
|   |-- dataset_exploration.ipynb
|   |-- 01_svm_rbf.ipynb
|   |-- 02_xgboost.ipynb
|   |-- 03_mlp.ipynb
|   |-- 04_cnn_spatial_head.ipynb
|   |-- 05_convnext_colab.ipynb
|   |-- 06_ensemble_svm_xgb_cnn.ipynb
|   `-- 07_deconvnet_reconstruction.ipynb  # solo estensione futura, non implementare ora
|-- artifacts/
|   |-- dataset_manifest.json
|   `-- splits_seed42.npz
`-- results/
    |-- dataset_exploration/
    |-- svm_rbf/
    |-- xgboost/
    |-- mlp/
    |-- cnn_spatial_head/
    |-- convnext/
    |-- ensemble/
    `-- deconvnet/
```

I notebook e le directory di output possono essere creati durante l'implementazione. I file originali `features_0.npy` e `features_1.npy` non devono mai essere modificati o sovrascritti. Tutti i dataset derivati e tutti i dati preprocessati devono essere salvati nella stessa directory `dataset/`, con nomi distinti e descrittivi.

## 5. Regole generali di implementazione

1. Usare sempre `pathlib.Path`. L'unico path assoluto Colab ammesso e la root Drive canonica dichiarata nella cella di setup; tutti gli altri path devono essere derivati da tale root.
2. Caricare i tensori con `numpy.load(..., mmap_mode="r", allow_pickle=False)`.
3. Non concatenare in RAM i due tensori spaziali completi.
4. Non modificare o sovrascrivere i file del dataset.
5. Impostare almeno i seed di `random`, NumPy e PyTorch.
6. Attivare le opzioni deterministiche di PyTorch quando ragionevolmente compatibili con le operazioni usate.
7. Registrare versioni di Python, NumPy, scikit-learn, XGBoost, PyTorch e CUDA.
8. Separare nettamente tuning/early stopping dalla valutazione finale sul test.
9. Non usare il test set per scegliere iperparametri, epoche, pesi dell'ensemble, calibrazione o soglia.
10. Usare `balanced accuracy` come unica metrica primaria di selezione su validation e salvare il modello migliore secondo tale metrica, non il modello dell'ultima epoca.
11. Non sovrascrivere silenziosamente risultati precedenti: ogni esecuzione deve avere un `run_id` o una directory chiaramente identificabile.
12. Ogni notebook deve poter essere rieseguito integralmente con `Restart Kernel and Run All`.
13. Ogni notebook deve rilevare automaticamente se e in esecuzione su Google Colab o in ambiente locale.
14. Non usare path Windows assoluti, lettere di unita o separatori hard-coded. Usare sempre `pathlib.Path` e una configurazione unica della root del progetto.
15. Ogni notebook deve contenere la cella condizionale di mount di Google Drive specificata in questo documento. Il mount viene eseguito quando l'ambiente Colab e rilevato e saltato automaticamente in locale.
16. Le dipendenze mancanti devono essere installabili tramite una cella iniziale Colab-safe, senza reinstallare pacchetti gia presenti inutilmente.
17. Il codice deve supportare fallback ordinato `CUDA -> CPU`; nessuna cella deve assumere che una GPU sia sempre disponibile.
18. Le operazioni specifiche di Windows, comandi PowerShell e path locali non devono comparire nella logica necessaria all'esecuzione dei notebook.
19. Non applicare data augmentation alle feature. Sono vietati flip, rotazioni, crop, rumore artificiale, MixUp, CutMix, channel dropout applicato all'input e qualsiasi trasformazione casuale dei campioni.
20. Dropout, stochastic depth, weight decay ed early stopping sono consentiti come regolarizzazione interna del modello: non sono data augmentation e non modificano i dati salvati.
21. Non usare SMOTE o altre tecniche di generazione sintetica/oversampling dei campioni. L'eventuale sbilanciamento deve essere gestito con metriche bilanciate o pesi di classe selezionati sul validation set.

### Compatibilita locale e Google Colab

La posizione persistente canonica del progetto su Google Drive e:

```text
/content/drive/MyDrive/Magistrale/Advanced_ML/progetto_aml
```

La posizione persistente canonica dei dataset originali e preprocessati e:

```text
/content/drive/MyDrive/Magistrale/Advanced_ML/progetto_aml/dataset
```

Ogni notebook, incluso `dataset_exploration.ipynb`, deve contenere all'inizio la seguente cella comune. La cella monta Drive, verifica dimensione e presenza dei due file originali, li copia nella VM Colab e configura separatamente lettura temporanea e scrittura persistente:

```python
from pathlib import Path
import importlib.util
import shutil

IS_COLAB = importlib.util.find_spec("google.colab") is not None

EXPECTED_FILES = {
    "features_0.npy": 2_167_931_008,
    "features_1.npy": 3_123_445_888,
}

if IS_COLAB:
    from google.colab import drive

    DRIVE_MOUNT = Path("/content/drive")

    if not (DRIVE_MOUNT / "MyDrive").exists():
        drive.mount(str(DRIVE_MOUNT), force_remount=False)

    DRIVE_PROJECT_ROOT = (
        DRIVE_MOUNT
        / "MyDrive"
        / "Magistrale"
        / "Advanced_ML"
        / "progetto_aml"
    )

    PERSISTENT_DATASET_DIR = DRIVE_PROJECT_ROOT / "dataset"

    if not PERSISTENT_DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory non trovata su Drive: "
            f"{PERSISTENT_DATASET_DIR}"
        )

    # Storage temporaneo e veloce della VM Colab.
    RUNTIME_PROJECT_ROOT = Path("/content/progetto_aml")
    RUNTIME_DATASET_DIR = RUNTIME_PROJECT_ROOT / "dataset"
    RUNTIME_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for filename, expected_size in EXPECTED_FILES.items():
        source = PERSISTENT_DATASET_DIR / filename
        destination = RUNTIME_DATASET_DIR / filename

        if not source.is_file():
            raise FileNotFoundError(
                f"File non trovato su Google Drive: {source}"
            )

        if source.stat().st_size != expected_size:
            raise RuntimeError(
                f"Dimensione non valida per {source}: "
                f"{source.stat().st_size} byte; "
                f"attesi {expected_size} byte"
            )

        copy_required = (
            not destination.exists()
            or destination.stat().st_size != expected_size
        )

        if copy_required:
            print(f"Copio {filename} da Drive alla VM Colab...")
            shutil.copy2(source, destination)
        else:
            print(
                f"{filename} gia presente nella VM: "
                "copia saltata."
            )

        if destination.stat().st_size != expected_size:
            raise RuntimeError(
                f"Copia incompleta: {destination}"
            )

    # I modelli leggono i due grandi file originali dalla VM veloce.
    DATASET_DIR = RUNTIME_DATASET_DIR

    # Dati preprocessati e output devono restare persistenti su Drive.
    PROCESSED_DATASET_DIR = PERSISTENT_DATASET_DIR
    ARTIFACTS_DIR = DRIVE_PROJECT_ROOT / "artifacts"
    RESULTS_DIR = DRIVE_PROJECT_ROOT / "results"

else:
    PROJECT_ROOT = Path.cwd()

    # Consente l'esecuzione anche quando il notebook parte da notebooks/.
    if not (PROJECT_ROOT / "dataset").is_dir():
        PROJECT_ROOT = PROJECT_ROOT.parent

    PERSISTENT_DATASET_DIR = PROJECT_ROOT / "dataset"
    DATASET_DIR = PERSISTENT_DATASET_DIR
    PROCESSED_DATASET_DIR = PERSISTENT_DATASET_DIR
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    RESULTS_DIR = PROJECT_ROOT / "results"

if not PERSISTENT_DATASET_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset directory non trovata: {PERSISTENT_DATASET_DIR}"
    )

PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Ambiente:", "Google Colab" if IS_COLAB else "Locale")
print("Dataset per lettura:", DATASET_DIR)
print("Dataset persistente:", PROCESSED_DATASET_DIR)
print("Artifacts:", ARTIFACTS_DIR)
print("Results:", RESULTS_DIR)
```

Il codice reale puo essere piu robusto, ma deve rispettare questi principi:

- ambiente rilevato automaticamente;
- root configurata in una sola cella;
- unico path assoluto Colab dichiarato nella cella di configurazione;
- nessun altro path assoluto incorporato nel resto del notebook;
- file originali letti da `DATASET_DIR`, che su Colab punta alla copia veloce nella VM;
- qualsiasi dataset derivato o file di dati preprocessati scritto esclusivamente in `PROCESSED_DATASET_DIR`;
- `PROCESSED_DATASET_DIR` punta alla cartella persistente Drive `progetto_aml/dataset` su Colab e alla cartella locale `dataset` fuori da Colab;
- artefatti scritti sotto `PROJECT_ROOT / "artifacts"` e `PROJECT_ROOT / "results"`;
- creazione delle directory mancanti con `mkdir(parents=True, exist_ok=True)`;
- selezione del device PyTorch tramite disponibilita effettiva di CUDA;
- AMP attivata solo su device compatibili;
- `num_workers` configurabile, con valore conservativo di default;
- salvataggio degli artefatti nello stesso formato su entrambe le piattaforme.

Quando eseguito su Colab, la cella copia sempre i due `.npy` originali nello storage locale della VM se non sono gia presenti con la dimensione corretta. La copia e una cache temporanea e viene ricreata in una nuova runtime. I notebook devono leggere `features_0.npy` e `features_1.npy` da `DATASET_DIR`, ma devono scrivere ogni dato preprocessato in `PROCESSED_DATASET_DIR`.

Non salvare mai il risultato finale del preprocessing soltanto sotto `/content`: verrebbe perso alla chiusura della runtime. Se per efficienza il file viene costruito prima nella VM, al termine deve essere copiato atomicamente in `PROCESSED_DATASET_DIR` e verificato per dimensione o hash.

### Contratto comune delle celle locale/Colab

Non devono esistere due versioni dello stesso notebook e non devono essere mantenute celle alternative da attivare manualmente per locale e Colab. Ogni `.ipynb` deve usare:

1. una sola cella iniziale condizionale per rilevamento ambiente, mount, copia e configurazione dei path;
2. celle identiche per il caricamento dei dati;
3. celle identiche per split e preprocessing;
4. celle identiche per la produzione degli output.

Soltanto la cella iniziale contiene il ramo `if IS_COLAB`. Tutto il codice successivo deve dipendere esclusivamente da:

```text
DATASET_DIR
PROCESSED_DATASET_DIR
ARTIFACTS_DIR
RESULTS_DIR
IS_COLAB
```

Significato delle directory:

| Variabile | Ambiente locale | Google Colab |
|---|---|---|
| `DATASET_DIR` | `project-root/dataset` | copia temporanea `/content/progetto_aml/dataset` |
| `PROCESSED_DATASET_DIR` | `project-root/dataset` | Drive `progetto_aml/dataset` |
| `ARTIFACTS_DIR` | `project-root/artifacts` | Drive `progetto_aml/artifacts` |
| `RESULTS_DIR` | `project-root/results` | Drive `progetto_aml/results` |

### Cella comune per il caricamento dei tensori originali

Dopo la cella iniziale, tutti i notebook che richiedono le feature map devono usare una cella equivalente a:

```python
import numpy as np

features_real = np.load(
    DATASET_DIR / "features_0.npy",
    mmap_mode="r",
    allow_pickle=False,
)

features_fake = np.load(
    DATASET_DIR / "features_1.npy",
    mmap_mode="r",
    allow_pickle=False,
)

assert features_real.shape == (1654, 1280, 16, 16)
assert features_fake.shape == (2383, 1280, 16, 16)
assert features_real.dtype == np.float32
assert features_fake.dtype == np.float32

print("Real:", features_real.shape, features_real.dtype)
print("Fake:", features_fake.shape, features_fake.dtype)
```

Questa cella non deve contenere rami specifici per Colab: `DATASET_DIR` e gia stato configurato correttamente dalla cella iniziale.

### Cella comune per il caricamento dei dati preprocessati

I notebook tabulari devono caricare le cache persistenti con una cella equivalente a:

```python
gap_features = np.load(
    PROCESSED_DATASET_DIR / "tabular_features_gap.npy",
    mmap_mode="r",
    allow_pickle=False,
)

mean_std_features = np.load(
    PROCESSED_DATASET_DIR / "tabular_features_mean_std.npy",
    mmap_mode="r",
    allow_pickle=False,
)
```

Il notebook puo caricare soltanto la rappresentazione richiesta dall'esperimento corrente, ma il path deve sempre essere derivato da `PROCESSED_DATASET_DIR`.

### Cella comune per il caricamento dello split

```python
splits = np.load(
    ARTIFACTS_DIR / "splits_seed42.npz",
    allow_pickle=False,
)

train_indices = splits["train_indices"]
validation_indices = splits["validation_indices"]
test_indices = splits["test_indices"]
```

I nomi delle chiavi devono essere identici in tutti i notebook. Se lo split usa chiavi differenti durante l'implementazione, aggiornare una sola volta il contratto e tutti i notebook.

### Cella comune per le directory di output

Ogni notebook di modello deve creare le directory degli output con codice indipendente dalla piattaforma:

```python
MODEL_NAME = "replace_with_model_name"
RUN_ID = "seed42"

RUN_DIR = RESULTS_DIR / MODEL_NAME / RUN_ID
FIGURES_DIR = RUN_DIR / "figures"

RUN_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
```

Metriche, configurazioni, predizioni, modelli e figure devono essere salvati usando soltanto `RUN_DIR` e `FIGURES_DIR`:

```python
metrics_df.to_csv(RUN_DIR / "metrics.csv", index=False)
fig.savefig(
    FIGURES_DIR / "loss_curve.png",
    dpi=200,
    bbox_inches="tight",
)
```

Non introdurre path Colab o locali nelle celle di training e valutazione.

### Salvataggio sicuro dei checkpoint PyTorch

Su Colab i checkpoint grandi o frequenti possono essere scritti prima nella VM e poi copiati nella directory persistente. La differenza deve essere nascosta in una funzione comune:

```python
def save_torch_checkpoint(checkpoint, filename, run_dir):
    final_path = run_dir / filename

    if IS_COLAB:
        temporary_path = Path("/content") / filename
        torch.save(checkpoint, temporary_path)
        shutil.copy2(temporary_path, final_path)
        temporary_path.unlink()
    else:
        torch.save(checkpoint, final_path)

    if not final_path.is_file():
        raise RuntimeError(
            f"Checkpoint non salvato correttamente: {final_path}"
        )

    return final_path
```

La funzione deve essere usata nello stesso modo in locale e Colab. Il checkpoint persistente finale deve sempre trovarsi sotto `RUN_DIR`. Per modelli piccoli, CSV, JSON e figure e consentita la scrittura diretta in `RUN_DIR`.

### Salvataggio comune dei dati preprocessati

I notebook che producono dati preprocessati devono usare sempre `PROCESSED_DATASET_DIR`, indipendentemente dalla piattaforma:

```python
output_path = (
    PROCESSED_DATASET_DIR
    / "tabular_features_mean_std.npy"
)

np.save(output_path, mean_std_features)
```

Su Colab questo salva in Google Drive; in locale salva nella directory `dataset` del progetto. Nessun dato preprocessato definitivo deve rimanere esclusivamente in `DATASET_DIR` quando questa variabile punta alla VM temporanea.

## 6. Split condiviso

Tutti i notebook devono usare gli stessi indici di train, validation e test, salvati in:

```text
artifacts/splits_seed42.npz
```

Split predefinito, se non sono disponibili gruppi:

- training: 70%;
- validation: 15%;
- test: 15%;
- stratificazione per classe;
- seed: 42.

Dimensioni attese con split casuale stratificato:

- train: circa 2826 campioni;
- validation: circa 605 campioni;
- test: circa 606 campioni.

Se esistono campioni correlati, usare invece uno split stratificato per gruppo. Tutte le varianti provenienti dalla stessa sorgente devono essere assegnate a un solo split.

Il notebook `dataset_exploration.ipynb` e il responsabile principale della creazione del manifest e dello split. Ogni notebook di modello deve comunque contenere una cella idempotente di caricamento e validazione che:

1. crea lo split solo se il file non esiste, usando esattamente la funzione definita nel notebook esplorativo;
2. altrimenti carica gli indici esistenti;
3. verifica che gli insiemi siano disgiunti;
4. verifica che la loro unione contenga tutti i campioni;
5. verifica distribuzione delle classi e range degli indici;
6. calcola e registra un hash stabile dello split.

Una volta prodotti risultati definitivi, lo split non deve essere rigenerato.

## 7. Notebook iniziale `dataset_exploration.ipynb`

Prima di implementare o addestrare i modelli deve essere creato ed eseguito:

```text
notebooks/dataset_exploration.ipynb
```

Questo notebook deve essere compatibile sia con esecuzione locale sia con Google Colab e deve rispettare le stesse regole di configurazione, portabilita, riproducibilita e salvataggio degli altri notebook.

### Obiettivi

Il notebook deve:

1. documentare struttura e qualita del dataset;
2. creare `artifacts/dataset_manifest.json`;
3. creare o validare `artifacts/splits_seed42.npz`;
4. produrre le rappresentazioni tabulari GAP e mean+std riutilizzabili dentro `dataset/`;
5. individuare duplicati, near-duplicate, outlier e possibili fonti di leakage;
6. misurare quanto le classi siano separabili attraverso statistiche semplici;
7. generare figure pronte per relazione e presentazione;
8. non addestrare o valutare i modelli finali.

### Regola contro il test leakage

Sono ammesse sull'intero dataset soltanto analisi strutturali che non guidano la scelta del modello, per esempio shape, dtype, memoria, NaN, Inf, duplicati e distribuzione numerica complessiva.

Dopo la creazione dello split, tutte le esplorazioni class-conditional che possono influenzare preprocessing, feature engineering, architetture o iperparametri devono usare esclusivamente il training set. Il validation set puo essere mostrato soltanto per controlli di distribuzione dichiarati; il test set deve rimanere nascosto fino alla valutazione finale dei modelli.

Non produrre PCA, UMAP, t-SNE, ranking dei canali o distribuzioni separate per classe usando il test set.

### Esplorativa 1: inventario e integrita dei file

Per ogni `.npy` registrare:

- nome e path relativo;
- dimensione in byte e GiB;
- versione del formato NPY;
- shape;
- dtype e byte order;
- ordine C/Fortran;
- numero di campioni;
- byte per campione;
- hash del file o, se troppo costoso, hash documentato dell'header e impronta a blocchi.

Verificare che le dimensioni dichiarate nell'header siano coerenti con la dimensione del file.

### Esplorativa 2: qualita numerica

Calcolare in streaming, senza caricare tutto in RAM:

- conteggio di NaN e valori infiniti;
- frazione di valori uguali a zero;
- minimo e massimo;
- media e deviazione standard globale;
- percentili robusti;
- eventuali valori estremi;
- distribuzione per campione di media, deviazione standard, minimo, massimo, norma L1 e norma L2.

Produrre istogrammi, box plot o violin plot delle statistiche per campione usando soltanto il training set per confronti tra classi.

### Esplorativa 3: bilanciamento delle classi

Mostrare:

- numero e percentuale di campioni per classe;
- baseline accuracy del classificatore maggioritario;
- conteggi per train, validation e test;
- eventuale distribuzione per gruppo o sorgente, se i metadati sono disponibili.

Salvare un grafico a barre con conteggi e percentuali.

### Esplorativa 4: duplicati e near-duplicate

Calcolare un hash stabile per ogni campione al fine di individuare duplicati binari esatti:

- dentro la stessa classe;
- tra classi diverse;
- tra split diversi.

Per i near-duplicate usare una rappresentazione ridotta, per esempio mean+std pooling seguito da PCA, quindi nearest neighbors o similarita coseno. Salvare le coppie piu simili con `sample_id`, distanza, classe e split.

Non eliminare automaticamente i campioni sospetti. Produrre una lista da ispezionare e documentare ogni eventuale esclusione.

### Esplorativa 5: pooling e statistiche per canale

Creare per ogni campione:

- media per ciascuno dei 1280 canali;
- deviazione standard per ciascuno dei 1280 canali;
- opzionalmente massimo e minimo per canale come analisi secondaria.

Sulla sola porzione di training calcolare:

- media e varianza per canale e classe;
- effect size standardizzato tra classi;
- ranking dei canali maggiormente discriminativi;
- ranking dei canali a varianza maggiore;
- percentuali di canali con effect size sopra soglie predefinite;
- eventuali test statistici con correzione per confronti multipli, chiarendo che significativita statistica non equivale a rilevanza predittiva.

Produrre almeno:

- distribuzione degli effect size;
- grafico dei migliori 20 canali;
- heatmap delle correlazioni dei canali selezionati;
- confronto per classe delle attivazioni dei canali principali.

### Esplorativa 6: struttura spaziale delle feature

Sulle feature `(1280, 16, 16)` produrre:

- mappa media dell'attivazione assoluta per posizione spaziale e classe;
- mappa della differenza tra classi calcolata soltanto sul training;
- mappa della varianza spaziale;
- heatmap di un piccolo insieme di canali rappresentativi;
- esempi di feature map per campioni selezionati in modo riproducibile.

Non tentare di visualizzare tutti i 1280 canali. Selezionare canali tramite varianza o effect size calcolati sul training set.

### Esplorativa 7: riduzione dimensionale

Applicare tecniche di riduzione dimensionale alla rappresentazione mean+std o global average pooling, mai al tensore flattened da 327680 dimensioni.

Analisi consigliate:

- PCA: varianza spiegata cumulativa e scatter delle prime componenti;
- t-SNE su un campione riproducibile o sull'intero training set, se sostenibile;
- UMAP opzionale, installato condizionalmente e con seed fissato.

Colorare i grafici per classe e, in una figura distinta, per gruppo/sorgente se disponibile. Non usare il test set nei grafici class-conditional.

### Esplorativa 8: outlier

Identificare possibili outlier tramite piu segnali:

- robust z-score delle statistiche globali;
- distanza nello spazio PCA;
- nearest-neighbor distance;
- Isolation Forest opzionale sul training set.

Salvare un CSV degli outlier candidati con motivazione e punteggio. Non rimuoverli automaticamente.

### Esplorativa 9: sanity check sui confondenti

Usando solo train e validation, addestrare un classificatore diagnostico molto semplice basato su:

```text
media globale, deviazione standard globale, minimo globale, massimo globale
```

Questo non e uno dei modelli finali. Serve a verificare se le classi sono separabili usando sole differenze globali. Riportare balanced accuracy e ROC-AUC su validation. Se il risultato e elevato, evidenziare il rischio che preprocessing, sorgente o pipeline di estrazione introducano un confondente.

E inoltre possibile eseguire una baseline diagnostica su global average pooling, sempre senza usare il test set.

### Esplorativa 10: prestazioni del caricamento

Misurare in modo leggero:

- tempo di apertura con memory mapping;
- throughput di lettura per batch;
- RAM usata indicativamente;
- differenze tra storage locale e Google Drive quando eseguito su Colab;
- batch size ragionevoli per i modelli spaziali.

Questa sezione deve evitare benchmark lunghi o che consumino inutilmente quote GPU.

### Cache e artefatti prodotti

Il notebook deve produrre almeno:

```text
artifacts/dataset_manifest.json
artifacts/splits_seed42.npz
dataset/tabular_features_gap.npy
dataset/tabular_features_mean_std.npy
dataset/spatial_channel_normalization.npz
dataset/sample_index.csv
results/dataset_exploration/summary.json
results/dataset_exploration/channel_statistics.csv
results/dataset_exploration/near_duplicates.csv
results/dataset_exploration/outlier_candidates.csv
results/dataset_exploration/figures/
```

`sample_index.csv` deve collegare ogni riga della cache a:

```text
sample_id,source_file,source_index,label,split
```

Le cache tabulari possono essere calcolate sull'intero dataset perche GAP e mean+std pooling sono trasformazioni deterministiche non apprese. Qualsiasi scaler, PCA, selezione di feature o trasformazione appresa deve invece essere fittata esclusivamente sul training set.

`dataset/spatial_channel_normalization.npz` deve contenere 1280 medie e 1280 deviazioni standard calcolate esclusivamente sui campioni di training e su tutte le rispettive posizioni spaziali. Deve includere anche split hash, epsilon usato e dtype.

Tutti i notebook che eseguono preprocessing devono salvare i dati preprocessati in `PROCESSED_DATASET_DIR`. Su Colab questa variabile punta a `/content/drive/MyDrive/Magistrale/Advanced_ML/progetto_aml/dataset`; in locale coincide con la cartella `dataset` del progetto. `ARTIFACTS_DIR` deve contenere soltanto metadati, split e manifest; `RESULTS_DIR` deve contenere modelli, metriche, predizioni e figure.

Il manifest deve contenere mapping delle classi, shape, dtype, conteggi, hash disponibili, seed, split hash e data/ora di generazione.

### Figure minime del notebook esplorativo

- `class_distribution.png`;
- `sample_global_statistics.png`;
- `channel_effect_size_distribution.png`;
- `top_channels.png`;
- `selected_channel_correlation.png`;
- `spatial_activation_by_class.png`;
- `spatial_difference_map.png`;
- `pca_explained_variance.png`;
- `pca_projection_train.png`;
- `tsne_or_umap_train.png`, se eseguita;
- `outlier_scores.png`;
- `diagnostic_baseline_roc.png`.

Tutte le figure devono seguire gli stessi requisiti grafici dei notebook di modello ed essere adatte alla relazione e alle slide.

## 8. Struttura obbligatoria di ogni notebook

Tutti i notebook devono contenere le seguenti sezioni, nello stesso ordine.

### 1. Titolo e obiettivo

- nome del modello;
- obiettivo dell'esperimento;
- input utilizzato;
- hardware previsto.

### 2. Setup e riproducibilita

- import;
- rilevamento automatico ambiente locale/Google Colab;
- esecuzione della cella comune di mount, copia e configurazione dei path;
- installazione condizionale delle sole dipendenze mancanti;
- configurazione dei seed;
- versioni delle librerie;
- rilevamento CPU/GPU;
- configurazione portabile dei path tramite `pathlib.Path`;
- configurazione centralizzata degli iperparametri.

### 3. Configurazione delle classi

- mapping esplicito file-label;
- nome della classe positiva;
- controllo che la classe positiva sia `fake` per precision, recall, F1 e PR-AUC.

### 4. Caricamento e validazione dei dati

- uso esclusivo di `DATASET_DIR` per i file originali e `PROCESSED_DATASET_DIR` per i dati preprocessati;
- nessun ramo locale/Colab nelle celle di caricamento successive al setup;
- memory mapping;
- shape e dtype;
- controlli NaN/Inf;
- distribuzione classi;
- stima memoria.

### 5. Split condiviso

- creazione o caricamento di `splits_seed42.npz`;
- verifiche di consistenza;
- riepilogo train/validation/test.

### 6. Preprocessing

- trasformazioni specifiche per il modello;
- fitting del preprocessing esclusivamente sul training set;
- salvataggio dei trasformatori necessari all'inferenza.

### 7. Definizione del modello

- architettura o configurazione;
- numero di parametri per le reti neurali;
- spazio di ricerca degli iperparametri.

### 8. Addestramento e selezione

- training;
- early stopping dove disponibile;
- selezione tramite validation;
- salvataggio del miglior modello.

### 9. Curve e diagnostica

- loss train/validation quando disponibile;
- accuracy train/validation quando disponibile;
- learning curve o validation curve per SVM;
- tempo e utilizzo hardware.

### 10. Valutazione finale

- metriche su train, validation e test;
- confusion matrix;
- ROC curve;
- precision-recall curve;
- reliability diagram, quando sono disponibili probabilita calibrate.

### 11. Salvataggio degli artefatti

- uso esclusivo di `RUN_DIR`, `FIGURES_DIR` e `PROCESSED_DATASET_DIR` come destinazioni astratte;
- nessun path locale o Colab hard-coded nelle celle di output;
- modello;
- configurazione;
- preprocessing;
- history;
- metriche;
- predizioni;
- figure.

### 12. Riepilogo

- tabella compatta delle metriche;
- migliori iperparametri;
- commento su overfitting e generalizzazione;
- path degli artefatti creati.

## 9. Rappresentazioni delle feature

### Regola generale sul preprocessing

Il preprocessing deve essere deterministico. Non deve essere applicata alcuna data augmentation, ne ai vettori tabulari ne alle feature map spaziali.

In particolare, non usare:

- flip orizzontali o verticali;
- rotazioni, crop o resize casuali;
- rumore gaussiano o perturbazioni casuali;
- MixUp o CutMix;
- mascheramento casuale di canali o regioni;
- SMOTE o campioni sintetici.

Le uniche trasformazioni ammesse sugli input sono pooling deterministico, conversione di dtype, standardizzazione con statistiche del training ed eventuale PCA fittata sul training. Dropout e stochastic depth possono essere usati dentro le reti come regolarizzazione del modello.

### Modelli tabulari

SVM RBF, XGBoost e MLP devono confrontare due rappresentazioni deterministiche.

Rappresentazione A, GAP:

```text
media per canale sulle 256 posizioni spaziali
= 1280 feature per campione
```

Per un campione `X` di shape `(1280, 16, 16)`:

```python
gap = X.mean(axis=(1, 2), dtype=np.float32)
```

Questa e la rappresentazione piu vicina al prototype pooling del paper, ma media l'intera immagine perche non sono disponibili le maschere degli oggetti.

Rappresentazione B, mean+std:

```text
mean per canale: 1280 valori
+
standard deviation per canale: 1280 valori
=
2560 feature per campione
```

Usare la deviazione standard della popolazione sulle 256 posizioni, quindi `ddof=0` o comportamento equivalente:

```python
mean = X.mean(axis=(1, 2), dtype=np.float32)
std = X.std(axis=(1, 2), ddof=0, dtype=np.float32)
mean_std = np.concatenate([mean, std]).astype(np.float32, copy=False)
```

L'ordine deve essere sempre `[mean_0, ..., mean_1279, std_0, ..., std_1279]` e deve essere registrato nel manifest.

Il calcolo deve avvenire in batch, senza caricare il dataset spaziale completo in RAM. Non applicare logaritmi, clipping o normalizzazioni per campione alle cache di base.

Mean+std e la rappresentazione tabulare principale. GAP deve essere eseguita come ablation study per SVM, XGBoost e MLP. Per ogni modello la variante da usare nel confronto finale e nell'ensemble viene scelta esclusivamente tramite validation, mai tramite test.

SVM e MLP devono usare uno standardizzatore fittato esclusivamente sul training set. XGBoost puo usare le feature non standardizzate. La specifica del pooling deve essere salvata in JSON per rendere riproducibile l'inferenza.

### Modelli spaziali

CNN spatial head e ConvNeXt devono utilizzare direttamente i tensori `(1280, 16, 16)` tramite un Dataset PyTorch basato su memory mapping.

Calcolare sul solo training set, per ogni canale `c`:

```text
channel_mean[c] = media su campioni train e posizioni 16x16
channel_std[c]  = deviazione standard su campioni train e posizioni 16x16
```

Applicare poi a train, validation e test:

```python
X_normalized = (X - channel_mean[:, None, None]) / (
    channel_std[:, None, None] + 1e-6
)
```

Usare le stesse statistiche salvate in `PROCESSED_DATASET_DIR / "spatial_channel_normalization.npz"` per CNN e ConvNeXt. Non calcolare statistiche separate per classe o split. Non normalizzare ogni campione indipendentemente e non applicare trasformazioni casuali alle feature map.

Su Windows partire con `num_workers=0` e aumentare solo dopo aver verificato che non ci siano copie eccessive del memory map. Usare `pin_memory=True` quando si addestra su GPU.

## 10. Modello 1: SVM RBF

Notebook: `notebooks/01_svm_rbf.ipynb`.

Pipeline:

```text
feature tensor
-> GAP oppure mean+std pooling
-> selezione degli indici train/validation/test
-> StandardScaler fittato esclusivamente sul train
-> PCA opzionale fittata esclusivamente sul train
-> SVM RBF
-> probabilita calibrata
```

Preprocessing obbligatorio:

1. caricare `PROCESSED_DATASET_DIR / "tabular_features_gap.npy"` o `PROCESSED_DATASET_DIR / "tabular_features_mean_std.npy"` mantenendo l'allineamento con `PROCESSED_DATASET_DIR / "sample_index.csv"`;
2. selezionare gli split tramite gli indici condivisi;
3. fittare `StandardScaler` solo su `X_train`;
4. trasformare train, validation e test con lo stesso scaler;
5. non applicare imputazione: il notebook deve fallire esplicitamente se trova NaN o Inf;
6. non usare oversampling, SMOTE o data augmentation;
7. mantenere i dati in `float32` dove possibile, lasciando alla pipeline scikit-learn eventuali conversioni interne.

La configurazione primaria deve partire senza PCA. PCA puo essere valutata soltanto come iperparametro opzionale con `n_components` pari a `256`, `512` o varianza spiegata `0.95`. Deve essere collocata dopo lo scaler dentro la pipeline, fittata solo sul training e selezionata tramite validation.

Spazio di ricerca iniziale:

```text
C: [0.1, 1, 10, 100]
gamma: ["scale", 1e-4, 5e-4, 1e-3]
class_weight: [null, "balanced"]
```

La ricerca deve usare solo training e validation. Se si usa cross-validation, questa deve avvenire dentro il training set.

La SVM RBF di scikit-learn non fornisce una vera storia della loss per epoca. Non fabbricare una curva di loss. Salvare invece:

- learning curve rispetto alla dimensione del training set;
- heatmap o validation curve di `C` e `gamma`;
- hinge loss finale per split;
- ROC e precision-recall;
- matrice di confusione.

Salvataggio richiesto:

```text
results/svm_rbf/<run_id>/best_model.joblib
```

Il file deve contenere pooling selezionato, scaler, eventuale PCA, SVM e componente di calibrazione necessaria all'ensemble. Se si usa `CalibratedClassifierCV`, evitare leakage e documentare la strategia di calibrazione.

## 11. Modello 2: XGBoost

Notebook: `notebooks/02_xgboost.ipynb`.

Input: GAP da 1280 feature e mean+std da 2560 feature, valutati in esperimenti separati. Mean+std e la configurazione primaria e GAP e l'ablation.

Preprocessing obbligatorio:

1. caricare la cache scelta e gli indici condivisi;
2. verificare dtype, valori finiti e allineamento dei `sample_id`;
3. usare direttamente le feature `float32`, senza `StandardScaler`;
4. non applicare PCA, normalizzazione per campione o selezione supervisionata prima del tuning;
5. non usare oversampling, SMOTE o data augmentation;
6. utilizzare gli eventuali pesi di classe solo come iperparametro, selezionato tramite validation.

Gli alberi non richiedono standardizzazione perche le divisioni dipendono dall'ordinamento e dalle soglie delle singole feature. Conservare le feature originali mantiene interpretabili le importanze di media e deviazione standard dei canali.

Spazio di ricerca iniziale:

```text
n_estimators: massimo 1000 con early stopping
max_depth: [3, 4, 5, 6]
learning_rate: [0.02, 0.05, 0.1]
subsample: [0.7, 0.85, 1.0]
colsample_bytree: [0.5, 0.75, 1.0]
min_child_weight: [1, 5, 10]
```

Usare `tree_method="hist"` per l'esecuzione locale. Limitare la ricerca in modo compatibile con CPU e RAM disponibili.

Salvare per ogni boosting round:

- train logloss;
- validation logloss;
- train classification error o accuracy;
- validation classification error o accuracy.

Salvataggi richiesti:

```text
results/xgboost/<run_id>/best_model.json
results/xgboost/<run_id>/model_bundle.joblib
```

Il modello nativo e preferito per portabilita. Il bundle puo contenere eventuale calibratore e metadati di preprocessing.

## 12. Modello 3: MLP

Notebook: `notebooks/03_mlp.ipynb`.

Input: GAP da 1280 feature e mean+std da 2560 feature, con dimensione di input parametrizzata. Mean+std e la configurazione primaria e GAP e l'ablation.

Preprocessing obbligatorio:

1. caricare la cache tabulare e lo split condiviso;
2. fittare `StandardScaler` esclusivamente su `X_train`;
3. trasformare validation e test con lo stesso scaler;
4. convertire gli array trasformati in tensori `float32`;
5. non applicare PCA nella configurazione primaria;
6. non usare oversampling, SMOTE o data augmentation;
7. non ricalcolare lo scaler tra le epoche o sui mini-batch.

Lo scaler deve essere salvato insieme al modello. LayerNorm e Dropout nei layer nascosti sono consentiti come regolarizzazione interna e non sostituiscono la standardizzazione dell'input.

Architettura iniziale:

```text
2560
-> Linear 512
-> LayerNorm
-> GELU
-> Dropout
-> Linear 128
-> LayerNorm
-> GELU
-> Dropout
-> Linear 1
```

Configurazione iniziale:

- loss: `BCEWithLogitsLoss`;
- optimizer: AdamW;
- learning rate: `1e-3`;
- batch size: 64 o 128;
- dropout: da 0.2 a 0.4;
- massimo 100 epoche;
- early stopping patience 12;
- scheduler: cosine decay o ReduceLROnPlateau.

Salvataggi richiesti:

```text
results/mlp/<run_id>/best_state_dict.pt
results/mlp/<run_id>/model_config.json
results/mlp/<run_id>/best_model_scripted.pt
```

Salvare sia `state_dict` e configurazione, sia una versione TorchScript quando la conversione e supportata.

## 13. Modello 4: CNN spatial head locale

Notebook: `notebooks/04_cnn_spatial_head.ipynb`.

Preprocessing obbligatorio:

1. aprire i due `.npy` originali da `DATASET_DIR` tramite memory mapping e recuperare ogni campione tramite `PROCESSED_DATASET_DIR / "sample_index.csv"`;
2. convertire il singolo campione in `float32` senza creare copie dell'intero dataset;
3. normalizzare ciascun canale usando esclusivamente `channel_mean` e `channel_std` calcolati sul training;
4. usare epsilon `1e-6` e le statistiche salvate in `PROCESSED_DATASET_DIR / "spatial_channel_normalization.npz"`;
5. restituire feature di shape `(1280, 16, 16)` e label scalare coerente con `0=real`, `1=fake`;
6. non applicare flip, crop, rotazioni, rumore, MixUp, CutMix o mascheramenti casuali;
7. non normalizzare ogni campione separatamente;
8. applicare AMP soltanto durante forward/backward su GPU, mantenendo le statistiche di normalizzazione in `float32`.

Il Dataset e il DataLoader non devono ricevere alcuna pipeline di augmentation. Shuffle e consentito soltanto per l'ordine dei campioni del training DataLoader; validation e test devono usare `shuffle=False`.

Architettura consigliata:

```text
Input 1280 x 16 x 16
-> Conv 1x1, 1280 -> 128
-> GroupNorm + GELU
-> 3 residual depthwise-separable blocks
-> channel attention
-> global average pooling + global max pooling
-> Linear 256 -> 64
-> GELU + Dropout
-> Linear 64 -> 1
```

Il modello deve rimanere compatibile con 6 GB di VRAM. Usare:

- batch size iniziale 8, poi provare 16;
- automatic mixed precision;
- AdamW;
- learning rate iniziale `3e-4`;
- massimo 80 epoche;
- early stopping patience 12;
- gradient clipping a 1.0;
- GroupNorm invece di BatchNorm quando il batch e piccolo.

Salvataggi richiesti:

```text
results/cnn_spatial_head/<run_id>/best_state_dict.pt
results/cnn_spatial_head/<run_id>/model_config.json
results/cnn_spatial_head/<run_id>/best_model_scripted.pt
```

Questo modello deve produrre probabilita o logit riutilizzabili dall'ensemble.

## 14. Modello 5: ConvNeXt su Google Colab

Notebook: `notebooks/05_convnext_colab.ipynb`.

Preprocessing obbligatorio:

1. usare gli stessi `PROCESSED_DATASET_DIR / "sample_index.csv"`, split e `PROCESSED_DATASET_DIR / "spatial_channel_normalization.npz"` della CNN spatial head;
2. copiare i file `.npy` nello storage locale Colab prima dell'addestramento, senza modificarne il contenuto;
3. aprire le copie locali tramite memory mapping;
4. applicare la stessa normalizzazione per canale della CNN, con statistiche calcolate esclusivamente sul training;
5. mantenere shape `(1280, 16, 16)` e dtype logico `float32` prima dell'eventuale autocast;
6. non applicare alcuna data augmentation o trasformazione casuale delle feature;
7. usare `shuffle=True` solo per il training DataLoader e `shuffle=False` per validation e test.

Usare esattamente lo stesso preprocessing spaziale per CNN e ConvNeXt e necessario per attribuire eventuali differenze di performance all'architettura anziche a trasformazioni diverse dei dati.

Architettura iniziale:

```text
Input 1280 x 16 x 16
-> Conv 1x1, 1280 -> 256
-> 3 ConvNeXt blocks a 16 x 16
-> downsample, 256 -> 384, 8 x 8
-> 3 ConvNeXt blocks
-> downsample, 384 -> 512, 4 x 4
-> 2 ConvNeXt blocks
-> LayerNorm
-> global average pooling
-> Dropout
-> Linear 512 -> 1
```

Obiettivo indicativo: 5-10 milioni di parametri. Non creare una rete eccessivamente grande rispetto ai 4037 esempi.

Configurazione iniziale:

- loss: `BCEWithLogitsLoss`;
- optimizer: AdamW;
- learning rate: `3e-4`;
- weight decay: da `1e-3` a `1e-2`;
- cosine scheduler con warm-up;
- massimo 100 epoche;
- early stopping patience 15;
- AMP;
- gradient accumulation per batch effettivo 32-64;
- dropout e stochastic depth.

Comportamento Colab obbligatorio:

1. eseguire la cella comune con `from google.colab import drive` e `drive.mount("/content/drive", force_remount=False)`;
2. usare come root persistente `/content/drive/MyDrive/Magistrale/Advanced_ML/progetto_aml`;
3. usare come directory dataset persistente la sottocartella `dataset`;
4. copiare facoltativamente i file `.npy` nello storage locale della sessione per accelerare le letture;
5. non trattare mai `/content` come destinazione persistente dei dati preprocessati;
6. rilevare automaticamente GPU e memoria disponibile;
7. adattare il batch size in configurazione;
8. salvare periodicamente checkpoint e history su Drive;
9. supportare la ripresa da checkpoint dopo interruzione della sessione.

Questi requisiti di portabilita si applicano anche agli altri notebook. La differenza e che ConvNeXt e progettato per essere eseguito principalmente su Colab, mentre SVM, XGBoost, MLP e CNN spatial head devono essere efficienti in locale ma restare eseguibili senza modifiche anche su Colab.

Salvataggi richiesti:

```text
results/convnext/<run_id>/best_state_dict.pt
results/convnext/<run_id>/last_checkpoint.pt
results/convnext/<run_id>/model_config.json
results/convnext/<run_id>/best_model_scripted.pt
```

## 15. Metriche comuni

Per ciascuno split salvare almeno:

- numero di campioni;
- loss, quando definita;
- accuracy;
- balanced accuracy, metrica primaria per selezione, tuning, early stopping e confronto dei modelli;
- precision della classe fake;
- recall/sensitivity della classe fake;
- specificity della classe real;
- F1 della classe fake;
- ROC-AUC;
- PR-AUC/average precision;
- Brier score quando sono disponibili probabilita;
- matrice di confusione;
- soglia decisionale;
- tempo di training;
- tempo di inferenza.

Schema minimo di `metrics.csv`:

```text
run_id,model,split,n_samples,loss,accuracy,balanced_accuracy,
precision,recall,specificity,f1,roc_auc,pr_auc,brier_score,
threshold,training_time_seconds,inference_time_seconds
```

Le metriche devono essere calcolate sempre con la stessa classe positiva e la stessa implementazione.

## 16. History e grafici

Per XGBoost, MLP, CNN e ConvNeXt salvare un `history.csv` con una riga per epoca o boosting round:

```text
step,train_loss,val_loss,train_accuracy,val_accuracy,learning_rate
```

Figure minime:

- `loss_curve.png`;
- `accuracy_curve.png`;
- `confusion_matrix_train.png`;
- `confusion_matrix_validation.png`;
- `confusion_matrix_test.png`;
- `roc_curve.png`;
- `precision_recall_curve.png`;
- `calibration_curve.png`, se applicabile.

Requisiti grafici:

- titolo descrittivo;
- legenda;
- label degli assi;
- griglia leggera;
- risoluzione almeno 150 DPI;
- layout adatto a essere inserito nelle slide;
- salvataggio prima di `plt.show()`.

Per SVM sostituire le curve per epoca con:

- `learning_curve.png`;
- `hyperparameter_heatmap.png`;
- metriche finali train/validation/test.

## 17. Predizioni salvate

Ogni notebook deve salvare le predizioni per train, validation e test con schema comune:

```text
sample_id,source_file,source_index,split,y_true,raw_score,probability,y_pred
```

Il campo `sample_id` deve essere stabile tra tutti i notebook. Un formato possibile e `features_0:123`.

Questi file servono a:

- costruire l'ensemble senza riaddestrare i modelli;
- confrontare gli errori campione per campione;
- eseguire bootstrap appaiato e test di McNemar;
- verificare l'allineamento tra modelli.

## 18. Ensemble SVM + XGBoost + CNN spatial head

Notebook: `notebooks/06_ensemble_svm_xgb_cnn.ipynb`.

L'ensemble deve contenere esclusivamente:

- SVM RBF;
- XGBoost;
- CNN spatial head.

MLP e ConvNeXt non devono essere componenti dell'ensemble. ConvNeXt e il termine di confronto principale.

Il notebook dell'ensemble non deve riaddestrare i tre modelli base. Deve caricare i migliori artefatti salvati dai notebook precedenti e verificare:

- stesso mapping delle classi;
- stesso hash dello split;
- stesso ordine dei campioni;
- assenza di predizioni mancanti;
- compatibilita delle versioni degli artefatti.

### Calibrazione

Le probabilita dei tre modelli devono essere confrontabili:

- SVM: Platt scaling o calibrazione cross-validated;
- XGBoost: Platt scaling o isotonic regression;
- CNN: temperature scaling sui logit.

La calibrazione puo essere appresa sul validation set. Questa scelta deve essere dichiarata; il test non deve essere usato.

### Varianti da confrontare

1. Soft voting uniforme:

```text
p = (p_svm + p_xgb + p_cnn) / 3
```

2. Soft voting pesato:

```text
p = w_svm * p_svm + w_xgb * p_xgb + w_cnn * p_cnn
```

con pesi non negativi e somma uguale a 1. Cercare i pesi sul validation set con griglia a passo 0.05 o metodo equivalente semplice e riproducibile.

La metrica di ottimizzazione obbligatoria e `balanced accuracy` calcolata sul validation set. Registrare anche il risultato con pesi uniformi.

La soglia decisionale deve essere scelta sul validation set e poi bloccata. Riportare anche le metriche con soglia 0.5.

### Analisi della diversita

Salvare:

- matrice di correlazione delle probabilita;
- matrice di correlazione degli errori;
- percentuale di accordo tra modelli;
- conteggio dei casi corretti esclusivamente da ciascun modello;
- diagramma di Venn o grafico equivalente degli errori, se leggibile.

### Bundle dell'ensemble

Creare un bundle portabile:

```text
results/ensemble/<run_id>/bundle/
|-- svm_pipeline.joblib
|-- xgboost_model.json
|-- xgboost_calibrator.joblib
|-- cnn_spatial_scripted.pt
|-- ensemble_config.json
`-- README.md
```

`ensemble_config.json` deve contenere almeno:

```json
{
  "components": ["svm_rbf", "xgboost", "cnn_spatial_head"],
  "weights": {
    "svm_rbf": null,
    "xgboost": null,
    "cnn_spatial_head": null
  },
  "threshold": null,
  "positive_class": "fake",
  "pooling": "channel_mean_and_std",
  "split_hash": null
}
```

I valori `null` devono essere sostituiti con i valori appresi sul validation set.

## 19. Confronto ensemble contro ConvNeXt

Il notebook dell'ensemble deve caricare anche le predizioni test di ConvNeXt, ma ConvNeXt non deve entrare nell'ensemble.

Creare una tabella finale con almeno:

- SVM RBF;
- XGBoost;
- CNN spatial head;
- ensemble uniforme;
- ensemble pesato;
- ConvNeXt.

Confrontare:

- tutte le metriche comuni;
- intervalli di confidenza bootstrap al 95%;
- bootstrap appaiato della differenza di balanced accuracy, ROC-AUC e PR-AUC;
- test di McNemar per gli errori binari di ensemble e ConvNeXt;
- tempo di training;
- tempo di inferenza;
- dimensione dei modelli;
- hardware utilizzato.

Non dichiarare un modello superiore se gli intervalli di confidenza e i test appaiati non supportano la conclusione. Se le differenze sono piccole, descrivere le performance come comparabili entro l'incertezza statistica.

## 20. DeconvNet per ricostruzione delle immagini

Notebook: `notebooks/07_deconvnet_reconstruction.ipynb`.

Stato corrente: non implementare questo notebook. Le immagini originali non sono disponibili e manca quindi il target supervisionato necessario. La sezione documenta esclusivamente una possibile estensione futura.

### Prerequisito obbligatorio

La ricostruzione supervisionata e possibile solo se sono disponibili le immagini originali accoppiate uno-a-uno con i tensori e se l'allineamento degli indici e verificabile.

Con le sole feature si puo addestrare un autoencoder delle feature, ma non una rete capace di verificare la ricostruzione delle immagini originali. Non presentare un feature autoencoder come ricostruzione d'immagine.

### Architettura indicativa per output 256 x 256

```text
Input 1280 x 16 x 16
-> Conv 1x1, 1280 -> 512
-> upsample 2x + Conv, 512 -> 256       # 32 x 32
-> upsample 2x + Conv, 256 -> 128       # 64 x 64
-> upsample 2x + Conv, 128 -> 64        # 128 x 128
-> upsample 2x + Conv, 64 -> 32         # 256 x 256
-> Conv 3x3, 32 -> 3
```

Preferire bilinear upsampling seguito da convoluzione rispetto a `ConvTranspose2d` per ridurre gli artefatti a scacchiera. Una variante con transposed convolution puo essere inclusa come ablation.

Senza skip connection e usando un unico layer di feature, e probabile ottenere struttura globale e colori approssimativi ma perdere dettagli fini.

### Loss indicativa

```text
total_loss = L1 + lambda_ssim * (1 - SSIM) + lambda_lpips * LPIPS
```

Non introdurre inizialmente una loss avversaria. Una GAN puo essere valutata solo dopo avere una baseline stabile e metriche quantitative affidabili.

### Metriche di ricostruzione

- MAE;
- MSE;
- PSNR;
- SSIM;
- LPIPS;
- griglie original/reconstruction/absolute error sul test set.

Usare lo stesso split dei classificatori, se gli indici delle immagini corrispondono alle feature.

## 21. Controlli contro leakage e bias

Prima di considerare valido un risultato verificare:

1. preprocessing identico per real e fake prima dell'estrazione delle feature;
2. stessa rete e stesso checkpoint di estrazione;
3. stessa risoluzione e normalizzazione;
4. nessuna immagine sorgente presente in split diversi;
5. scaler fittato solo sul train;
6. nessuna scelta effettuata osservando il test;
7. soglie e pesi scelti solo sul validation;
8. predizioni allineate tramite `sample_id` e non soltanto tramite ordine delle righe.
9. nessuna data augmentation, trasformazione casuale degli input, SMOTE o oversampling applicato in alcun notebook.
10. SVM e MLP usano scaler fittati soltanto sul training; XGBoost usa feature non standardizzate; CNN e ConvNeXt condividono le stesse statistiche per canale del training.
11. ogni notebook contiene la cella condizionale `from google.colab import drive` e `drive.mount("/content/drive", force_remount=False)`.
12. su Colab, la root persistente e `/content/drive/MyDrive/Magistrale/Advanced_ML/progetto_aml`.
13. tutti i dataset derivati e i dati preprocessati sono salvati dentro `PROCESSED_DATASET_DIR` e non soltanto nella VM temporanea.

Come sanity check, confrontare i modelli con un classificatore che usa soltanto quattro statistiche globali: media, deviazione standard, minimo e massimo. Se tale modello e molto accurato, discutere esplicitamente la possibilita di dataset bias.

## 22. Criteri di accettazione

Un notebook e completo soltanto se:

- si esegue dall'inizio alla fine con kernel pulito;
- lo stesso file si esegue sia in locale sia su Google Colab senza modificare il codice delle celle;
- rileva automaticamente l'ambiente e usa una sola cella di configurazione dei percorsi;
- non contiene path Windows assoluti o dipendenze obbligatorie da PowerShell;
- installa su Colab soltanto le dipendenze mancanti e supporta CPU quando CUDA non e disponibile;
- usa lo split condiviso e ne verifica l'hash;
- non modifica il dataset;
- non sovrascrive `features_0.npy` o `features_1.npy`;
- salva ogni dato preprocessato o dataset derivato dentro `PROCESSED_DATASET_DIR`;
- non applica data augmentation o generazione sintetica di campioni;
- salva e ricarica tutti i componenti del preprocessing necessari all'inferenza;
- salva il miglior modello ricaricabile;
- ricarica il modello salvato e verifica che le predizioni coincidano entro tolleranza;
- salva metriche per train, validation e test;
- salva le predizioni per campione;
- salva tutti i grafici previsti;
- registra configurazione, seed e versioni;
- non usa il test per tuning o early stopping;
- riporta chiaramente limiti e possibili bias.

L'ensemble e completo soltanto se puo eseguire inferenza partendo dal proprio bundle, senza riaddestrare i componenti.

## 23. Ordine di implementazione

1. Definire mapping real/fake e, se disponibili, gruppi.
2. Implementare ed eseguire `dataset_exploration.ipynb`.
3. Creare manifest, cache tabulare e split condiviso dal notebook esplorativo.
4. Revisionare duplicati, near-duplicate, outlier e possibili confondenti prima di addestrare i modelli.
5. Implementare SVM RBF e verificare l'intero contratto degli artefatti.
6. Implementare XGBoost.
7. Implementare MLP.
8. Implementare CNN spatial head e validare il consumo VRAM locale.
9. Implementare ConvNeXt e adattare il notebook a Colab/Drive.
10. Verificare e allineare tutte le predizioni salvate.
11. Implementare calibrazione ed ensemble.
12. Confrontare ensemble e ConvNeXt con analisi statistica appaiata.
13. Non implementare attualmente la DeconvNet; rivalutarla soltanto se in futuro saranno disponibili immagini originali accoppiate.
14. Generare tabelle e figure finali per la relazione e la presentazione.

## 24. Decisioni ancora aperte

- eventuale disponibilita di ID di gruppo;
- strategia finale di calibrazione;
- numero di seed da usare nei risultati definitivi: raccomandati almeno 3 per le reti neurali.

Le immagini originali non sono disponibili. La DeconvNet di ricostruzione non fa quindi parte dell'implementazione corrente e rimane soltanto una possibile estensione futura qualora fossero recuperate immagini target correttamente allineate alle feature.
