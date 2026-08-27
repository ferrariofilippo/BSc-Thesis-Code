# Mesh Adapt

## Utilizzo
### Pre Work
1. Aprire il [Google Sheet](https://docs.google.com/spreadsheets/d/1cD-qDc7uT6IMTMIjpP1LOpsteqq0uyMXMu6NCYIGWbc/edit?usp=sharing) con le configurazioni dei problemi
2. Configurare i parametri di input nel foglio `Dashboard`
3. Aprire `Mesh_Adapt/freefem/main.edp`, cercare `func f0` e definire le funzioni usate nel punto `2.` 

### Avvio
1. Esportare il foglio `export` dal *Google Sheet* come `.CSV` (**ATTENZIONE:** impostare il foglio in inglese, altrimenti la virgola "separatore decimale", fa casino con la virgola "separatore del csv")
2. Inserire il file esportato nella cartella `Mesh_Adapt/data_in/` (rinominarlo `data.csv`)
3. 
```bash
cd Mesh_Adapt
python ./sim-runner.py
```

# Dettagli Specifici

## File

- `data_in/`
  - `data.csv`: CSV contenente i parametri di input
  - `data-evaluation.csv`: CSV contenente i parametri di input per i problemi di valutazione del modello
- `data_out/`
  - `res.csv`: CSV contenente i risultati delle simulazioni
  - `res-evaluation.csv`: CSV contenente i risultati delle simulazioni per i problemi di valutazione del modello
  - `meshes/`: Mesh generate dallo script
  - `meshes-evaluation/`: Mesh per la valutazione del modello generate dallo script
- `freefem/`
  - `Adaptation/`: Script freefem per adattazione delle mesh
  - `main.edp`: Script freefem principale. Riceve in input i parametri per la simulazione. Crea la mesh e la adatta. Salva i risultati.
- `sim-runner.py`: Script python per lanciare automaticamente le simulazioni

## Script FreeFem
- **Codice:** `Mesh_Adapt/freefem/main.edp`

### Parametri in Input
```python
-id,          # Id della simulazione
-mukind,      # Tipo del coefficiente di diffusione € { "Costante", "Funzione i" con i = 1... }
-mu,          # Intensità del coefficiente di diffusione
-bkind1,      # Tipo della componente 1 del coefficiente di trasporto € { "Costante", "Funzione i" con i = 1... }
-bI1,         # Intensità della componente 1 del coefficiente di trasporto
-bkind2,      # Tipo della componente 2 del coefficiente di trasporto € { "Costante", "Funzione i" con i = 1... }
-bI2,         # Intensità della componente 2 del coefficiente di trasporto
-sigmakind,   # Tipo del coefficiente di reazione € { "Costante", "Funzione i" con i = 1... }
-sigmaI,      # Intensità del coefficiente di reazione
-fkind,       # Tipo di forzante € { "Costante", "Funzione i" con i = 1... }
-fI,          # Intensità della forzante
-l1bc,        # Tipo di condizione al bordo per il lato L1 € { Dirichlet, Neumann, Robin }
-l1bcfun,     # Tipo di funzione usata come BC per il lato L1 € { "Costante", "Funzione i" con i = 1... }
-l1bcI,       # Intensità della funzione usata come BC per il lato L1
-l2bc,        # Tipo di condizione al bordo per il lato L2 € { Dirichlet, Neumann, Robin }
-l2bcfun,     # Tipo di funzione usata come BC per il lato L2 € { "Costante", "Funzione i" con i = 1... }
-l2bcI,       # Intensità della funzione usata come BC per il lato L2
-l3bc,        # Tipo di condizione al bordo per il lato L3 € { Dirichlet, Neumann, Robin }
-l3bcfun,     # Tipo di funzione usata come BC per il lato L3 € { "Costante", "Funzione i" con i = 1... }
-l3bcI,       # Intensità della funzione usata come BC per il lato L3
-l4bc,        # Tipo di condizione al bordo per il lato L4 € { Dirichlet, Neumann, Robin }
-l4bcfun,     # Tipo di funzione usata come BC per il lato L4 € {"Costante", "Funzione i" con i = 1... }
-l4bcI,       # Intensità della funzione usata come BC per il lato L4
-toll,        # Tolleranza
-nbvx,        # Numero massimo di vertici
-alpha,       # Parametro per modificare le funzioni
-beta,        # Parametro per modificare le funzioni
-l13s,        # Lunghezza dei lati L1 e L3
-l24s,        # Lunghezza dei lati L2 e L4
-nx,          # Numeri di vertici iniziali in direzione x
-ny,          # Numero di vertici iniziali in direzione y
-hmin,        # Dimensione minima degli elementi         
-noPlot,      # 1 se non si vogliono vedere i grafici 
```

### Parametri Configurabili (statici)
```freefem
string DefultFunction = "Costante"; // Valore di default per le funzioni
string DefaultBC = "Dirichlet";     // Valore di default per le condizioni al bordo

real DefaultMagnitude = 1.0;        // Valore di default per le intensità

real DefaultToll = 5.0e-3;          // Valore di default della tolleranza
real DefaltNbvx = 500000;           // Valore di default di nbvx
int DefaultNx = 100;                // Valore di default di nx
int DefaultNy = 100;                // Valore di default di ny
int DefaultL13 = 1;                 // Valore di default di l13s
int DefaultL24 = 1;                 // Valore di default di l24s

real DefaultAlpha = 1;              // Velocità angolare di default di x
real DefaultBeta = 1;               // Velocità angolare di default di y

int DefaultNoPlot = 1;              // Non mostrare i grafici di default

int MinIterations = 5;              // Numero minimo di iterazioni per l'adattamento
int DefaultIterations = 30;         // Numero massimo di iterazioni per l'adattamento
real DefaultConvergenceThreshold = 0.01;  // Soglia di stagnazione per il numero di elementi della mesh
```

### Output
Lo script, per ogni lancio, stampa questo output
```csv
ID_SIMULAZIONE,NUMERO_ITERAZIONI,NUMERO_ELEMENTI,H_MIN,H_MAX,NUMERO_VERTICI,VARIANZA_ERRORE,STIMA_ERRORE
```

## Script Python
- **Version Python:** `3.13.5`
- **Codice:** `Mesh_Adapt/sim-runner.py`

### Requisiti
```bash
python3 -m pip install subprocess; python3 -m pip install time
```

### Altre indicazioni
Lo script prende autonomamente i parametri di input ed esegue le simulazioni.
