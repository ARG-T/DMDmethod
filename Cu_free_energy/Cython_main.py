from c_free_energy import FreeEnergy
import time


def culc_lat(temperature):
    lat_p = 3.646
    temperature = temperature
    FE = FreeEnergy(lat_p, temperature)
    result = FE.main_loop()
    return result


if __name__ == "__main__":
    s = time.time()
    result = culc_lat(500)
    print(result)
    print(time.time()-s)
