from c_free_energy import FreeEnergy
import time


if __name__ == "__main__":
    s = time.time()
    FE = FreeEnergy()
    FE.main_loop()
    print("time", time.time()-s)