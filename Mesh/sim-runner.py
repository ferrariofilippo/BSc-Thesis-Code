############################################# LIBRARIES ############################################
import subprocess
import time
from threading import Lock, BoundedSemaphore, Thread

############################################# SETTINGS #############################################
START_INDEX = 1
END_INDEX = 1361
N_THREADS = 14

INPUT_PARAMS_FILE = "./data_in/data.csv"
# INPUT_PARAMS_FILE = "./data_in/data-evaluation.csv"
CONTAINS_HEADER = True

OUTPUT_CSV_FILE = "./data_out/res.csv"
# OUTPUT_CSV_FILE = "./data_out/res-evaluation.csv"

FREE_FEM_SCRIPT = "./freefem/main.edp"

DEBUG = False
NO_PLOT = True

############################################# CORE LOGIC ###########################################
index_to_args_map = [
    "-id",          # Id della simulazione
    "-mukind",      # Tipo del coefficiente di diffusione € { "Costante", "Funzione i" con i = 1... }
    "-mu",          # Intensità del coefficiente di diffusione
    "-bkind1",      # Tipo della componente 1 del coefficiente di trasporto € { "Costante", "Funzione i" con i = 1... }
    "-bI1",         # Intensità della componente 1 del coefficiente di trasporto
    "-bkind2",      # Tipo della componente 2 del coefficiente di trasporto € { "Costante", "Funzione i" con i = 1... }
    "-bI2",         # Intensità della componente 2 del coefficiente di trasporto
    "-sigmakind",   # Tipo del coefficiente di reazione € { "Costante", "Funzione i" con i = 1... }
    "-sigmaI",      # Intensità del coefficiente di reazione
    "-fkind",       # Tipo di forzante € { "Costante", "Funzione i" con i = 1... }
    "-fI",          # Intensità della forzante
    "-l1bc",        # Tipo di condizione al bordo per il lato L1 € { Dirichlet, Neumann, Robin }
    "-l1bcfun",     # Tipo di funzione usata come BC per il lato L1 € { "Costante", "Funzione i" con i = 1... }
    "-l1bcI",       # Intensità della funzione usata come BC per il lato L1
    "-l2bc",        # Tipo di condizione al bordo per il lato L2 € { Dirichlet, Neumann, Robin }
    "-l2bcfun",     # Tipo di funzione usata come BC per il lato L2 € { "Costante", "Funzione i" con i = 1... }
    "-l2bcI",       # Intensità della funzione usata come BC per il lato L2
    "-l3bc",        # Tipo di condizione al bordo per il lato L3 € { Dirichlet, Neumann, Robin }
    "-l3bcfun",     # Tipo di funzione usata come BC per il lato L3 € { "Costante", "Funzione i" con i = 1... }
    "-l3bcI",       # Intensità della funzione usata come BC per il lato L3
    "-l4bc",        # Tipo di condizione al bordo per il lato L4 € { Dirichlet, Neumann, Robin }
    "-l4bcfun",     # Tipo di funzione usata come BC per il lato L4 € {"Costante", "Funzione i" con i = 1... }
    "-l4bcI",       # Intensità della funzione usata come BC per il lato L4
    "-toll",        # Tolleranza
    "-nbvx",        # Numero massimo di vertici
    "-alpha",       # Parametro per modificare le funzioni
    "-beta",        # Parametro per modificare le funzioni
    "-l13s",        # Lunghezza dei lati L1 e L3
    "-l24s",        # Lunghezza dei lati L2 e L4
    "-nx",          # Numeri di vertici iniziali in direzione x
    "-ny",          # Numero di vertici iniziali in direzione y
    "-hmin",        # Dimensione minima degli elementi         
    "-noPlot",      # 1 se non si vogliono vedere i grafici 
]

write_lock = Lock()
pool_semaphore = BoundedSemaphore(value=N_THREADS)

def save_res(res):
    write_lock.acquire()
    with open(OUTPUT_CSV_FILE, 'at+') as f:
        f.write(res)

    write_lock.release()

def run_script(params):
    args = [
        "freefem++", FREE_FEM_SCRIPT, 
        '-v', '0', 
        '-noPlot', '1' if NO_PLOT else '0',
    ]

    for i in range(len(params)):
        args.append(index_to_args_map[i])
        args.append(str(params[i]))

    if DEBUG:
        print(args)

    res = subprocess.run(args, capture_output=True, text=True)
    save_res(res.stdout)

    if DEBUG:
        print(res.stdout)
        print(res.stderr)

    pool_semaphore.release()

def read_csv():
    problems = []
    with open(INPUT_PARAMS_FILE, 'rt') as f:
        if CONTAINS_HEADER:
            f.readline()
        
        row = f.readline()
        while row != None and row != "":
            fields = row.strip('\n').split(',')
            row = f.readline()
            sampleid = 0

            try:
                sampleid = int(fields[0])
            except:
                continue

            if sampleid < START_INDEX:
                continue
            if (sampleid >= END_INDEX):
                break

            problems.append(fields)
            
    return problems

if __name__ == '__main__':
    threads = []

    print('--- Reading CSV: Started')
    samples = read_csv()
    print('--- Reading CSV: Done')

    print('--- Adaptation: Started')
    cnt = len(samples)
    for i in range(len(samples)):
        pool_semaphore.acquire()

        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time()))
        print(f' - Sample: {samples[i][0]}\tProcessing {i}/{cnt}\tTime: {ts}')

        th = Thread(target=run_script, args=[samples[i]])
        threads.append(th)
        th.start()

    for th in threads:
        th.join()

    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time()))

    print(f' - Processed {cnt}/{cnt}\tTime: {ts}')
    print('--- Adaptation: Done')
