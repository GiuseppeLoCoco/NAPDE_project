from mpi4py import MPI
import sys, subprocess, os, time
from os import getpid, path

# Printing helper functions 
RED = "\033[1;37;31m%s\033[0m"
BLUE = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__	

def remove_killvanDANA(directory):
    try:
        os.remove(path.join(directory, "killvanDANA"))
    except:
        pass

def remove_complete(directory):
    try:
        os.remove(path.join(directory, "complete"))
    except:
        pass

# MPI utility mappato su mpi4py
class MPI_Manage:
    def __init__(self):
        self.mpi_comm = MPI.COMM_WORLD
        self.my_rank  = self.mpi_comm.Get_rank()
        self.size     = self.mpi_comm.Get_size()

    def get_rank(self):
        return self.my_rank

    def set_barrier(self):
        self.mpi_comm.Barrier()

    def get_communicator(self): 
        return self.mpi_comm

    def Max(self, x):
        return self.mpi_comm.allreduce(x, op=MPI.MAX)

    def Min(self, x):
        return self.mpi_comm.allreduce(x, op=MPI.MIN)

    def Sum(self, x):
        return self.mpi_comm.allreduce(x, op=MPI.SUM)       

def create_counters(i):
    return [0 for _ in range(i)]

def reset_counter(j, *args):
    if args:
        for x in args: j[x] = 0
    else:
        for i in range(len(j)): j[i] = 0    

def update_counter(j, *args):
    if args:
         for x in args: j[x] += 1         
    else:
        for i in range(len(j)): j[i] += 1    

def getMemoryUsage(rss=True):
    mypid = str(getpid())
    rss_val = "rss" if rss else "vsz"
    process = subprocess.Popen(['ps', '-o', rss_val, mypid], stdout=subprocess.PIPE)
    out, _ = process.communicate()
    mymemory = out.split()[1]
    return eval(mymemory) / 1024

class MemoryUsage:
    def __init__(self, s):
        self.memory = 0
        self.memory_vm = 0
        self(s)

    def __call__(self, s, verbose=False):
        self.prev = self.memory
        self.prev_vm = self.memory_vm
        comm = MPI.COMM_WORLD
        self.memory = comm.allreduce(getMemoryUsage(), op=MPI.SUM)
        self.memory_vm = comm.allreduce(getMemoryUsage(False), op=MPI.SUM)
        if verbose and comm.Get_rank() == 0:
            print(BLUE % f'{s:26s}  {int(self.memory - self.prev):10d} MB {int(self.memory):10d} MB {int(self.memory_vm - self.prev_vm):10d} MB {int(self.memory_vm):10d} MB')

# Sostituto leggero del Timer di dolfin
class Timer:
    def __init__(self, name):
        self.name = name
        self.t_start = 0.0
    def start(self):
        self.t_start = time.time()
    def stop(self):
        pass # Può essere esteso per loggare o accumulare i tempi

timer_total = Timer("Total_run_time")
timer_dt    = Timer("Time_step_timer")
timer_s1    = Timer("Predict tentative velocity step")
timer_s2    = Timer("Pressure correction step")
timer_s3    = Timer("Velocity correction step")
timer_s4    = Timer("Energy conservation step")
timer_s5    = Timer("Solid momentum eq. step")
timer_s6    = Timer("Lagrange multiplier/fictitious force step")
timer_s7    = Timer("Solid temperature's lagrange multiplier step")
timer_si    = Timer("Delta_fx_interpolation")
timer_sm    = Timer("Move solid mesh")
timer_sr    = Timer("Remesh solid mesh")