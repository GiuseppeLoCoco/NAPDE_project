# NAPDE_project

> *"Moving geometries without remeshing: assessment and comparison of immersed methods for CFD"*

---

## Struttura della cartella `Project`

All'interno della cartella `Project/` il codice è organizzato nelle seguenti sottocartelle:

- **`domain_settings/`**: contiene i file necessari per le varie operazioni sulle mesh (generazione, raffinamento, gestione delle condizioni al contorno) e per la definizione geometrica degli ostacoli (es. sfere, cilindri).
- **`Experiments/`**: raccoglie gli script per eseguire gli esperimenti con il meccanismo di buffer, utili a testare e validare i metodi immersi nel recupero dello stato della soluzione.
- **`Solvers/`**: contiene le implementazioni dei vari solver per le equazioni di Navier-Stokes e Stokes (metodi conformi, Brinkman penalization, RIIS, DLM).
- **`user_inputs/`**: racchiude la definizione e la configurazione dei parametri predefiniti di simulazione e delle opzioni per i solutori.
- **`Utils/`**: raccoglie i moduli ausiliari, gli strumenti di post-processing, le soluzioni esatte (MMS) e il codice dedicato alla generazione dei grafici e dei plot dei risultati.
- **`validation/`**: contiene gli script dedicati alle analisi numeriche e di benchmark:
  - `convergence_analysis.py`: svolge l'analisi dell'ordine di convergenza spaziale/temporale dei metodi.
  - `test_L2_penalization.py`: esegue il test della penalizzazione $L^2$ (Brinkman) rispetto alla variazione del parametro resistivo, in conformità con i risultati di riferimento della letteratura/paper.
