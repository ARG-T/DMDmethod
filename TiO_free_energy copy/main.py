from culc_free_energy import FreeEnergy
import time


def culc_lat(temperature):
    temperature = temperature
    FE = FreeEnergy(temperature)
    result = FE.main_loop()
    return result


if __name__ == "__main__":
    s = time.time()
    result = culc_lat(300)
    print("result = ", result)
    print("time is ", time.time()-s)