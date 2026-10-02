# CNN spatial head: spiegazione dell'architettura

Questo documento descrive l'architettura implementata nel notebook `notebooks/04_cnn_spatial_head.ipynb`. La rete non riceve un'immagine RGB, ma feature map già estratte da un modello a monte. Il suo obiettivo è classificare tali feature map nelle classi **real** (`0`) e **fake** (`1`).

## Schema generale

L'input ha forma:

\[
[B,\;1280,\;16,\;16]
\]

dove `B` è la dimensione del batch, 1280 è il numero di canali e `16 x 16` è la griglia spaziale. L'architettura è:

```text
Input normalizzato [B, 1280, 16, 16]
        |
        v
Stem: Conv 1x1 1280 -> 128 + GroupNorm + GELU
        |
        v
3 ResidualDepthwiseBlock, in serie
        |
        v
Channel attention
        |
        v
Global average pooling e global max pooling, in parallelo
        |
        v
Concatenazione: 128 + 128 = 256 valori
        |
        v
Classification head: Linear 256 -> 64 -> 1
        |
        v
Logit binario (sigmoid in inferenza -> probabilità fake)
```

| Fase | Operazione | Shape output | Ruolo principale |
|---|---|---:|---|
| Input | Feature map normalizzata | `[B, 1280, 16, 16]` | Rappresentazione fornita dal modello a monte |
| Stem | Conv `1x1`, `1280 -> 128` + GroupNorm + GELU | `[B, 128, 16, 16]` | Compressione e rielaborazione dei canali |
| Blocchi residuali | 3 blocchi depthwise-separable | `[B, 128, 16, 16]` | Estrazione di pattern spaziali e mescolamento dei canali |
| Channel attention | Pooling + MLP `128 -> 16 -> 128` + sigmoid | `[B, 128, 16, 16]` | Ripesatura adattiva dei canali |
| Global pooling | Average e max, in parallelo | Due vettori `[B, 128]` | Sintesi spaziale media e dei picchi |
| Concatenazione | Average + max | `[B, 256]` | Fusione di informazioni complementari |
| Classificatore | `Linear 256 -> 64 -> 1` | `[B]` | Decisione binaria |

La griglia spaziale resta `16 x 16` fino al pooling globale finale.

## Lo stem

Nel notebook lo stem è definito come:

```python
self.stem = nn.Sequential(
    nn.Conv2d(1280, 128, 1, bias=False),
    nn.GroupNorm(8, 128),
    nn.GELU(),
)
```

In una CNN, lo *stem* è il blocco iniziale: riceve l'input e lo converte nella rappresentazione interna su cui lavorerà il resto della rete. Il termine richiama il "tronco" della rete, prima dei blocchi principali e della testa di classificazione.

Lo stem non serve soltanto a ridurre dimensionalità. Svolge tre funzioni:

1. **Compressione dei canali.** Riduce i canali da 1280 a 128, cioè di un fattore 10, senza modificare la griglia `16 x 16`.
2. **Combinazione e selezione delle feature.** Impara quali combinazioni delle 1280 feature in ingresso sono utili per il compito.
3. **Preparazione dell'attivazione.** GroupNorm stabilizza la distribuzione delle attivazioni e GELU introduce non linearità.

Il numero di valori di attivazione per campione passa da:

\[
1280 \cdot 16 \cdot 16 = 327\,680
\]

a:

\[
128 \cdot 16 \cdot 16 = 32\,768.
\]

La riduzione rende più economici i blocchi successivi, ma è una proiezione appresa e non una semplice eliminazione di valori.

## Convoluzione pointwise `1x1`

Una convoluzione `1x1` non osserva le posizioni spaziali vicine. In ciascuna posizione `(h,w)` riceve il vettore dei canali e applica una trasformazione lineare appresa:

\[
\mathbf{y}_{h,w} = W\mathbf{x}_{h,w} + \mathbf{b}.
\]

Nello stem:

\[
\mathbf{x}_{h,w}\in\mathbb{R}^{1280},
\qquad
\mathbf{y}_{h,w}\in\mathbb{R}^{128}.
\]

Quindi, per ognuna delle 256 posizioni della griglia, trasforma un vettore di 1280 feature in un vettore di 128 feature. Una pointwise convolution è efficace per:

- cambiare il numero di canali;
- combinare i canali;
- apprendere nuove feature come combinazioni di feature esistenti;
- contenere il costo computazionale.

Non estrae invece pattern locali nello spazio: questa è la funzione delle convoluzioni `3x3` presenti nei blocchi successivi.

## GroupNorm

La rete usa:

```python
nn.GroupNorm(8, 128)
```

I 128 canali sono divisi in 8 gruppi da 16 canali. Per ogni campione e per ogni gruppo, GroupNorm calcola media e varianza sulle dimensioni canale e spaziali; quindi normalizza:

\[
\hat{x} = \frac{x - \mu_{\text{gruppo}}}
{\sqrt{\sigma^2_{\text{gruppo}} + \epsilon}}.
\]

Infine applica due parametri appresi per canale:

\[
y = \gamma\hat{x} + \beta.
\]

GroupNorm non dipende dalle statistiche degli altri esempi del batch. È quindi particolarmente indicata quando il batch è piccolo, come nel notebook (`batch_size = 8`), e ha comportamento coerente fra training e inferenza. Al contrario, BatchNorm dipende dalle statistiche del batch e può diventare meno stabile con batch piccoli.

## GELU

GELU è la funzione di attivazione:

\[
\operatorname{GELU}(x) = x\Phi(x),
\]

dove \(\Phi\) è la funzione di distribuzione cumulativa della normale standard.

Intuitivamente, GELU è un filtro morbido: attenua molto i valori negativi, lascia passare quasi completamente quelli positivi e rende graduale la transizione in prossimità di zero. A differenza di ReLU, non tronca bruscamente tutti i valori negativi a zero.

## I tre blocchi residuali depthwise

La rete costruisce i blocchi con:

```python
self.blocks = nn.Sequential(
    *[ResidualDepthwiseBlock() for _ in range(3)]
)
```

I tre `ResidualDepthwiseBlock` sono quindi **in serie, cioè in cascata**:

\[
x \rightarrow B_1(x) \rightarrow B_2(\cdot) \rightarrow B_3(\cdot).
\]

Ogni blocco conserva la shape `[B, 128, 16, 16]`. Le tre convoluzioni depthwise `3x3` successive permettono alla rete di combinare informazione in un intorno spaziale teorico fino a circa `7x7` della feature map originale.

Ogni blocco è implementato così:

```python
outputs = self.activation(self.norm_a(self.depthwise(inputs)))
outputs = self.norm_b(self.pointwise(outputs))
return self.activation(inputs + outputs)
```

In forma matematica:

\[
F(x)=\operatorname{GN}_2\left(
\operatorname{PW}_{1\times1}
\left(
\operatorname{GELU}(\operatorname{GN}_1(\operatorname{DW}_{3\times3}(x)))
\right)
\right),
\]

\[
y=\operatorname{GELU}(x+F(x)).
\]

### Depthwise Conv2d `3x3`

Il layer è:

```python
nn.Conv2d(128, 128, 3, padding=1, groups=128)
```

Con `groups=128`, ogni canale è filtrato indipendentemente con un proprio kernel `3x3`. Il canale 1 usa solo il canale 1, il canale 2 usa solo il canale 2, e così via. La depthwise convolution estrae quindi pattern locali spaziali, come configurazioni di attivazioni in celle vicine della feature map.

Il `padding=1` garantisce che la dimensione spaziale resti `16x16`.

Una convoluzione standard `3x3` da 128 canali a 128 canali richiederebbe:

\[
128\cdot128\cdot3\cdot3 = 147\,456
\]

pesi. La depthwise convolution richiede invece:

\[
128\cdot3\cdot3 = 1\,152
\]

pesi. Tuttavia, da sola non combina i canali: per questo è seguita da una pointwise convolution.

### Pointwise Conv2d del blocco

Il layer:

```python
nn.Conv2d(128, 128, 1, bias=False)
```

combina i canali nella stessa posizione spaziale. Dopo che la depthwise conv ha analizzato lo spazio canale per canale, la pointwise conv mette in relazione le feature ottenute nei diversi canali.

La sequenza depthwise `3x3` + pointwise `1x1` è detta **depthwise-separable convolution**:

1. la depthwise convolution modella i rapporti nello spazio;
2. la pointwise convolution modella i rapporti fra canali.

Nel blocco il pointwise layer non riduce dimensionalità, perché esegue `128 -> 128`; rielabora invece la rappresentazione mescolando i canali.

### Residual add: relazione con ResNet

L'operazione:

```python
inputs + outputs
```

segue lo stesso principio dei collegamenti residui di ResNet. Invece di imparare direttamente una trasformazione completa \(H(x)\), il blocco impara una correzione residua \(F(x)\):

\[
H(x)=x+F(x).
\]

Il ramo principale calcola \(F(x)\), mentre lo *skip connection* trasporta direttamente l'input \(x\). Questo facilita il flusso del gradiente, consente di apprendere più facilmente trasformazioni vicine all'identità e aiuta a evitare il degrado dell'addestramento nelle reti profonde.

All'interno di ogni blocco i due rami sono concettualmente paralleli:

\[
\text{input }x
\begin{cases}
\rightarrow \text{ramo identità},\\
\rightarrow \text{depthwise + pointwise}=F(x)
\end{cases}
\rightarrow x+F(x).
\]

Non è presente una convoluzione di proiezione nello skip path perché input e output hanno già la stessa shape. Rispetto a una ResNet classica, il principio residuo è uguale, ma il ramo trasformativo usa convoluzioni depthwise-separable e non convoluzioni standard.

## Channel attention

La channel attention è:

```python
self.attention = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Conv2d(128, 16, 1),
    nn.GELU(),
    nn.Conv2d(16, 128, 1),
    nn.Sigmoid(),
)
```

ed è applicata tramite:

```python
outputs = outputs * self.attention(outputs)
```

È un meccanismo simile a **Squeeze-and-Excitation (SE)**.

### AdaptiveAvgPool2d(1): squeeze

Per ogni canale calcola la media sulle 256 posizioni spaziali:

\[
s_c = \frac{1}{16\cdot16}
\sum_{h=1}^{16}\sum_{w=1}^{16}x_{c,h,w}.
\]

Il tensore passa da `[B, 128, 16, 16]` a `[B, 128, 1, 1]`. Riduce quindi le dimensioni spaziali, ma non il numero di canali. Il termine *adaptive* indica che il layer produrrebbe comunque un output `1x1` anche se la dimensione di input fosse diversa.

### Bottleneck `128 -> 16 -> 128`

Le due convoluzioni `1x1`, applicate a feature map `1x1`, funzionano come un piccolo MLP sui canali:

\[
128 \rightarrow 16 \rightarrow 128.
\]

Il passaggio `128 -> 16` è una riduzione interna della dimensionalità con rapporto di compressione 8. La rete deve costruire una descrizione compatta delle relazioni fra canali; poi il layer `16 -> 128` restituisce un peso per ciascun canale.

### Sigmoid e moltiplicazione elemento per elemento

La sigmoid genera un gate per canale:

\[
g_c\in(0,1).
\]

La moltiplicazione elemento per elemento produce:

\[
\tilde{x}_{c,h,w}=g_c\cdot x_{c,h,w}.
\]

Lo stesso peso \(g_c\) viene applicato a tutte le posizioni spaziali del canale `c` tramite broadcasting. La rete può così attenuare canali poco utili per quello specifico esempio e conservare maggiormente quelli informativi. Non è un residual add: qui la feature map viene modulata moltiplicativamente.

## Global pooling parallelo

Dopo l'attenzione, il notebook applica due pool in parallelo:

```python
pooled = torch.cat([
    self.avg_pool(outputs).flatten(1),
    self.max_pool(outputs).flatten(1),
], dim=1)
```

\[
[B,128,16,16]
\rightarrow
\begin{cases}
\text{Global Average Pooling} \rightarrow [B,128],\\
\text{Global Max Pooling} \rightarrow [B,128]
\end{cases}
\rightarrow
\text{concatenazione} \rightarrow [B,256].
\]

Il global average pooling calcola, per ogni canale:

\[
a_c=\frac{1}{HW}\sum_{h,w}x_{c,h,w},
\]

e misura quanto una feature è mediamente presente nella mappa. Il global max pooling calcola invece:

\[
m_c=\max_{h,w}x_{c,h,w},
\]

e conserva la risposta più forte della feature, anche se localizzata in una sola posizione.

La concatenazione usa entrambe le informazioni. Il vettore passa da 128 a 256 componenti, ma la dimensionalità totale è comunque drasticamente inferiore alla feature map precedente: da `128 x 16 x 16 = 32.768` valori a 256.

## Classification head

La testa finale è:

```python
self.head = nn.Sequential(
    nn.Linear(256, 64),
    nn.GELU(),
    nn.Dropout(0.30),
    nn.Linear(64, 1),
)
```

Qui avviene una riduzione progressiva:

\[
256\rightarrow64\rightarrow1.
\]

- `Linear(256, 64)` combina le statistiche globali dei canali;
- `GELU` introduce non linearità;
- `Dropout(0.30)` azzera casualmente il 30% delle attivazioni durante il training, contribuendo a limitare l'overfitting;
- `Linear(64, 1)` produce un singolo **logit**.

Il logit non è ancora una probabilità. Durante l'inferenza il notebook applica la sigmoid:

\[
p(\text{fake})=\sigma(\text{logit}).
\]

## Riepilogo: riduzione dimensionale ed estrazione di feature

| Componente | Riduce dimensionalità? | Estrae/riformula feature? |
|---|---|---|
| Stem `1x1`, `1280 -> 128` | Sì, nei canali | Sì, combina i canali in nuove feature |
| Depthwise `3x3` | No | Sì, estrae pattern spaziali locali per canale |
| Pointwise `1x1`, `128 -> 128` | No | Sì, combina le feature fra canali |
| Residual add | No | Preserva l'input e aggiunge una correzione appresa |
| Attention `128 -> 16 -> 128` | Sì, solo nel bottleneck interno | Sì, stima l'importanza dei canali |
| Adaptive average/max pooling | Sì, elimina lo spazio `16x16 -> 1x1` | Riassume presenza media e risposta massima per canale |
| Classification head `256 -> 64 -> 1` | Sì | Combina le feature per la classificazione binaria |

In sintesi, lo stem comprime e riorganizza le feature; i tre blocchi residuali, in cascata, estraggono pattern spaziali; l'attenzione decide quali canali privilegiare; i pooling sintetizzano la griglia spaziale; il classificatore finale produce la decisione binaria.
